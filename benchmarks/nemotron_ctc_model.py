import io, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.asr import normalize_asr_text, word_error_rate
from exp2_asr_generalization.test_ofp_scaled_dev import load_dev_utterances

torch.set_grad_enabled(False)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d2 = x.shape[-1] // 2
    return torch.cat((-x[..., d2:], x[..., :d2]), dim=-1)

def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, Dh); cos, sin: (1, 1, T, r_dim)
    r_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :r_dim], q[..., r_dim:]
    k_rot, k_pass = k[..., :r_dim], k[..., r_dim:]
    q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)

class RoPECache(nn.Module):
    def __init__(self, head_dim: int, partial_rotary_factor: float = 0.5, base: float = 10000.0):
        super().__init__()
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[None, None, :, :]
        return emb.cos().to(dtype), emb.sin().to(dtype)

class NemotronTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        ln1 = self.norm1(x)
        qkv = self.w_qkv(ln1)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_partial_rope(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn_out)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x

class Nemotron35TranscribeCTC(nn.Module):
    def __init__(self, ckpt_path: str = '/dev/shm/wsp_audio/asr_data/nemotron_3_5_perception_ctc.pt', tok_path: str = '/dev/shm/wsp_audio/asr_data/nemotron_alignment_tokenizer.json'):
        super().__init__()
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.register_buffer('mel_fb', sd['perception.preprocessor.featurizer.fb'][0].float())  # (128, 257)
        self.register_buffer('stft_win', sd['perception.preprocessor.featurizer.window'].float())  # (400,)

        # 32-layer ASR encoder (d_model=1280, n_heads=10, ffn=5120)
        self.pre_encode = nn.Linear(1024, 1280, bias=False)
        self.pre_encode.weight.data.copy_(sd['perception.encoder.asr_encoder.pre_encode.proj.weight'].float())
        self.embed_norm = nn.LayerNorm(1280, eps=1e-5)
        self.embed_norm.weight.data.copy_(sd['perception.encoder.asr_encoder.embed_norm.weight'].float())
        self.embed_norm.bias.data.copy_(sd['perception.encoder.asr_encoder.embed_norm.bias'].float())

        self.enc_rope = RoPECache(head_dim=128, partial_rotary_factor=0.5, base=10000.0)
        self.enc_layers = nn.ModuleList([NemotronTransformerLayer(1280, 10, 5120) for _ in range(32)])
        for i in range(32):
            p = f'perception.encoder.asr_encoder.layers.{i}.'
            lyr = self.enc_layers[i]
            lyr.norm1.weight.data.copy_(sd[p+'norm1.weight'].float())
            lyr.norm1.bias.data.copy_(sd[p+'norm1.bias'].float())
            lyr.w_qkv.weight.data.copy_(sd[p+'attn.w_qkv.weight'].float())
            lyr.q_norm.weight.data.copy_(sd[p+'attn.q_norm.weight'].float())
            lyr.q_norm.bias.data.copy_(sd[p+'attn.q_norm.bias'].float())
            lyr.k_norm.weight.data.copy_(sd[p+'attn.k_norm.weight'].float())
            lyr.k_norm.bias.data.copy_(sd[p+'attn.k_norm.bias'].float())
            lyr.out_proj.weight.data.copy_(sd[p+'attn.out_proj.weight'].float())
            lyr.out_proj.bias.data.copy_(sd[p+'attn.out_proj.bias'].float())
            lyr.norm2.weight.data.copy_(sd[p+'norm2.weight'].float())
            lyr.norm2.bias.data.copy_(sd[p+'norm2.bias'].float())
            lyr.fc1.weight.data.copy_(sd[p+'ffn.net.0.weight'].float())
            lyr.fc1.bias.data.copy_(sd[p+'ffn.net.0.bias'].float())
            lyr.fc2.weight.data.copy_(sd[p+'ffn.net.3.weight'].float())
            lyr.fc2.bias.data.copy_(sd[p+'ffn.net.3.bias'].float())

        self.enc_final_norm = nn.LayerNorm(1280, eps=1e-5)
        self.enc_final_norm.weight.data.copy_(sd['perception.encoder.asr_encoder.final_norm.weight'].float())
        self.enc_final_norm.bias.data.copy_(sd['perception.encoder.asr_encoder.final_norm.bias'].float())

        # 2-layer CTC Timestamp Decoder (d_model=1280, n_heads=16, ffn=640)
        self.upsample = nn.ConvTranspose1d(1280, 1280, kernel_size=2, stride=2)
        self.upsample.weight.data.copy_(sd['ctc_timestamp.decoder.upsample.weight'].float())
        self.upsample.bias.data.copy_(sd['ctc_timestamp.decoder.upsample.bias'].float())

        self.dec_embed_norm = nn.LayerNorm(1280, eps=1e-5)
        self.dec_embed_norm.weight.data.copy_(sd['ctc_timestamp.decoder.transformer.embed_norm.weight'].float())
        self.dec_embed_norm.bias.data.copy_(sd['ctc_timestamp.decoder.transformer.embed_norm.bias'].float())

        self.dec_rope = RoPECache(head_dim=80, partial_rotary_factor=0.5, base=10000.0)
        self.dec_layers = nn.ModuleList([NemotronTransformerLayer(1280, 16, 640) for _ in range(2)])
        for i in range(2):
            p = f'ctc_timestamp.decoder.transformer.layers.{i}.'
            lyr = self.dec_layers[i]
            lyr.norm1.weight.data.copy_(sd[p+'norm1.weight'].float())
            lyr.norm1.bias.data.copy_(sd[p+'norm1.bias'].float())
            lyr.w_qkv.weight.data.copy_(sd[p+'attn.w_qkv.weight'].float())
            lyr.q_norm.weight.data.copy_(sd[p+'attn.q_norm.weight'].float())
            lyr.q_norm.bias.data.copy_(sd[p+'attn.q_norm.bias'].float())
            lyr.k_norm.weight.data.copy_(sd[p+'attn.k_norm.weight'].float())
            lyr.k_norm.bias.data.copy_(sd[p+'attn.k_norm.bias'].float())
            lyr.out_proj.weight.data.copy_(sd[p+'attn.out_proj.weight'].float())
            lyr.out_proj.bias.data.copy_(sd[p+'attn.out_proj.bias'].float())
            lyr.norm2.weight.data.copy_(sd[p+'norm2.weight'].float())
            lyr.norm2.bias.data.copy_(sd[p+'norm2.bias'].float())
            lyr.fc1.weight.data.copy_(sd[p+'ffn.net.0.weight'].float())
            lyr.fc1.bias.data.copy_(sd[p+'ffn.net.0.bias'].float())
            lyr.fc2.weight.data.copy_(sd[p+'ffn.net.3.weight'].float())
            lyr.fc2.bias.data.copy_(sd[p+'ffn.net.3.bias'].float())

        self.dec_final_norm = nn.LayerNorm(1280, eps=1e-5)
        self.dec_final_norm.weight.data.copy_(sd['ctc_timestamp.decoder.transformer.final_norm.weight'].float())
        self.dec_final_norm.bias.data.copy_(sd['ctc_timestamp.decoder.transformer.final_norm.bias'].float())
        self.residual_scale = float(sd['ctc_timestamp.decoder.residual_scale'].item())

        self.ctc_head = nn.Linear(1280, 23863, bias=True)
        self.ctc_head.weight.data.copy_(sd['ctc_timestamp.decoder.decoder_layers.0.weight'].squeeze(-1).float())
        self.ctc_head.bias.data.copy_(sd['ctc_timestamp.decoder.decoder_layers.0.bias'].float())

        # Load BPE vocabulary
        tok = json.load(open(tok_path))
        vocab = tok['model']['vocab']
        self.id_to_piece = {int(v): k for k, v in vocab.items()}
        self.blank_id = 23862

    def extract_mel(self, wav: torch.Tensor) -> tuple[torch.Tensor, int]:
        # wav: (1, L)
        L = wav.size(1)
        wav_pre = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], dim=1)
        stft = torch.stft(
            wav_pre, n_fft=512, hop_length=160, win_length=400,
            window=self.stft_win, return_complex=True, pad_mode='constant', center=True
        )
        mag_sq = stft.abs().pow(2)
        mel = torch.log((self.mel_fb @ mag_sq) + 2**-24).permute(0, 2, 1)  # (1, T_mel, 128)
        n_valid = L // 160
        mel = mel[:, :n_valid, :]
        # Per-feature normalization over valid frames
        mean = mel.mean(dim=1, keepdim=True)
        centered = mel - mean
        std = (centered.pow(2).sum(dim=1, keepdim=True) / max(1, n_valid - 1)).sqrt() + 1e-5
        normed = centered / std
        pad = (-n_valid) % 8
        if pad > 0:
            normed = F.pad(normed, (0, 0, 0, pad))
        return normed, n_valid

    def forward(self, wav: torch.Tensor, return_features: bool = False):
        normed_mel, _ = self.extract_mel(wav)
        B, T_pad, _ = normed_mel.shape
        stacked = normed_mel.reshape(B, T_pad // 8, 1024)
        x = self.embed_norm(self.pre_encode(stacked))
        cos_e, sin_e = self.enc_rope(x.size(1), x.device, x.dtype)
        enc_Penult = x
        for idx, lyr in enumerate(self.enc_layers):
            if idx == 31:
                enc_Penult = x
            x = lyr(x, cos_e, sin_e)
        asr_last = self.enc_final_norm(x)

        # Upsample 2x and run 2-layer CTC decoder
        u = self.upsample(asr_last.transpose(1, 2)).transpose(1, 2)
        h = self.dec_embed_norm(u)
        cos_d, sin_d = self.dec_rope(h.size(1), h.device, h.dtype)
        dec_l0 = h
        for idx, lyr in enumerate(self.dec_layers):
            h = lyr(h, cos_d, sin_d)
            if idx == 0:
                dec_l0 = h
        dec_normed = self.dec_final_norm(h)
        dec_out = u + self.residual_scale * dec_normed
        logits = self.ctc_head(dec_out)
        if return_features:
            # Upsample enc_Penult to match 2*T_enc frame rate
            return logits, dec_out, dec_normed, u, F.layer_norm(dec_l0, (1280,))
        return logits

    def decode_ctc(self, logits_2d: torch.Tensor) -> str:
        ids = logits_2d.argmax(dim=-1).tolist()
        pieces = []
        prev = -1
        for idx in ids:
            if idx != prev:
                if idx != self.blank_id and idx in self.id_to_piece:
                    p = self.id_to_piece[idx]
                    if not (p.startswith('<') and p.endswith('>')):
                        pieces.append(p)
            prev = idx
        text = ''.join(pieces).replace('\u2581', ' ')
        return normalize_asr_text(text)

if __name__ == '__main__':
    dev = torch.device('cuda:0')
    m = Nemotron35TranscribeCTC().to(dev).eval()
    n_p = sum(p.numel() for p in m.parameters())
    print(f'Loaded Nemotron-3.5-Transcribe 34-Layer BPE-CTC Model: {n_p/1e6:.2f}M params!')

    for d_name, col in [('dev_clean', 'text'), ('dev_other', 'text'), ('tedlium_dev', 'text'), ('mls_dev', 'transcript')]:
        utts = load_dev_utterances(d_name, col, max_utts=100)
        refs, hyps = [], []
        for wav, ref in utts:
            wt = torch.from_numpy(wav).unsqueeze(0).to(dev)
            logits = m(wt)
            hyp = m.decode_ctc(logits[0])
            refs.append(ref); hyps.append(hyp)
        wer = word_error_rate(refs, hyps) * 100.0
        print(f'  {d_name:12s} (100 utts): WER = {wer:5.2f}% | sample hyp: {hyps[0][:65]}', flush=True)
