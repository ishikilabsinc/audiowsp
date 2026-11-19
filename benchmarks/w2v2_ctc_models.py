"""Standalone PyTorch Wav2Vec2ForCTC Models & Factorized Low-Rank Compression Modules.

Supports:
  1. `facebook/wav2vec2-base-960h` (`94.40M` params, 12 Post-LN layers)
  2. `facebook/wav2vec2-large-960h` (`315.46M` params, 24 Post-LN layers)
  3. `facebook/wav2vec2-large-960h-lv60-self` (`315.47M` params, 24 Pre-LN layers)
plus in-place factorized low-rank layer replacement (`FactorizedLinear` and `TuckerPosConvEmbed`)
for `OT-SVD`, `StandardSVD`, `ASVD`, `Wanda`, and `Magnitude` compression baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedLinear(nn.Module):
    """Rank-`r` factorized linear projection `y = (x @ V_r^T) @ U_r^T + b` with `(D_in + D_out)*r + D_out` params."""

    def __init__(self, U_r: torch.Tensor, V_r: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        D_out, r = U_r.shape
        r2, D_in = V_r.shape
        assert r == r2
        self.in_features = D_in
        self.out_features = D_out
        self.rank = r
        self.V_proj = nn.Linear(D_in, r, bias=False)
        self.U_proj = nn.Linear(r, D_out, bias=bias is not None)
        self.V_proj.weight.data.copy_(V_r)
        self.U_proj.weight.data.copy_(U_r)
        if bias is not None:
            self.U_proj.bias.data.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U_proj(self.V_proj(x))


class ConvLayerGroupNorm(nn.Module):
    def __init__(self, in_d: int, out_d: int, kernel: int, stride: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_d, out_d, kernel_size=kernel, stride=stride, bias=bias)
        self.layer_norm = nn.GroupNorm(num_groups=out_d, num_channels=out_d, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.layer_norm(self.conv(x)))


class ConvLayerNoNorm(nn.Module):
    def __init__(self, in_d: int, out_d: int, kernel: int, stride: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_d, out_d, kernel_size=kernel, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x))


class ConvLayerLayerNorm(nn.Module):
    def __init__(self, in_d: int, out_d: int, kernel: int, stride: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_d, out_d, kernel_size=kernel, stride=stride, bias=bias)
        self.layer_norm = nn.LayerNorm(out_d, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer_norm(x.transpose(-2, -1)).transpose(-2, -1)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, norm_mode: str = "group", conv_bias: bool = False):
        super().__init__()
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        layers: List[nn.Module] = []
        for idx, (k, s) in enumerate(zip(kernels, strides)):
            in_c = 1 if idx == 0 else 512
            if norm_mode == "layer":
                layers.append(ConvLayerLayerNorm(in_c, 512, k, s, bias=conv_bias))
            else:
                if idx == 0:
                    layers.append(ConvLayerGroupNorm(in_c, 512, k, s, bias=conv_bias))
                else:
                    layers.append(ConvLayerNoNorm(in_c, 512, k, s, bias=conv_bias))
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lyr in self.conv_layers:
            x = lyr(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 768):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(x))


class PosConvEmbed(nn.Module):
    def __init__(self, embed_dim: int = 768, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        out = self.conv(x.transpose(1, 2))
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return F.gelu(out.transpose(1, 2))


class FastPosConvEmbed(nn.Module):
    """Pre-materialized or Mode-3 Tucker compressed Positional Convolution."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, groups: int = 16):
        super().__init__()
        C_out, C_in_g, K = weight.shape
        self.conv = nn.Conv1d(C_in_g * groups, C_out, kernel_size=K, padding=K // 2, groups=groups)
        self.conv.weight.data.copy_(weight)
        self.conv.bias.data.copy_(bias)
        self.remove = 1 if K % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x.transpose(1, 2))
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return F.gelu(out.transpose(1, 2))


class TuckerPosConvEmbed(nn.Module):
    """Mode-3 Kernel-SVD compressed Positional Conv (`5.17x` smaller)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, rank_t: int = 24, groups: int = 16):
        super().__init__()
        C, C_g, K = weight.shape
        W_mat = weight.reshape(C * C_g, K).float()
        U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)
        r = min(rank_t, S.numel())
        W_spatial = (U[:, :r] * S[:r].unsqueeze(0)).reshape(C, C_g, r)
        Basis_K = Vh[:r, :]  # (r, K)

        self.groups = groups
        self.rank_t = r
        self.K = K
        self.remove = 1 if K % 2 == 0 else 0
        self.register_buffer("basis_k", Basis_K.unsqueeze(1).contiguous())  # (r, 1, K)
        self.register_buffer("w_spatial", W_spatial.contiguous())  # (C, C_g, r)
        self.register_buffer("bias", bias.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B*C, 1, T)
        B, T, C = x.shape
        x_t = x.transpose(1, 2).reshape(B * C, 1, T)
        conv_k = F.conv1d(x_t, self.basis_k, padding=self.K // 2)  # (B*C, r, T_pad)
        if self.remove > 0:
            conv_k = conv_k[:, :, : -self.remove]
        T_out = conv_k.size(2)
        G = self.groups
        C_g = C // G
        conv_k = conv_k.view(B, G, C_g, self.rank_t, T_out)
        w_sp = self.w_spatial.view(G, C_g, C_g, self.rank_t)
        out = torch.einsum("bgirt,goir->bgot", conv_k, w_sp).reshape(B, C, T_out)
        out = out + self.bias.view(1, C, 1)
        return F.gelu(out.transpose(1, 2))


class Attention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj: nn.Module = nn.Linear(embed_dim, embed_dim)
        self.v_proj: nn.Module = nn.Linear(embed_dim, embed_dim)
        self.q_proj: nn.Module = nn.Linear(embed_dim, embed_dim)
        self.out_proj: nn.Module = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 768, ffn_dim: int = 3072):
        super().__init__()
        self.intermediate_dense: nn.Module = nn.Linear(embed_dim, ffn_dim)
        self.output_dense: nn.Module = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class LinearAdapterFFN(nn.Module):
    """Rank-`r` linear adapter replacing the 4096-dim FFN in deep semantic layers (`64x` smaller)."""

    def __init__(self, U_r: torch.Tensor, V_r: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.adapter = FactorizedLinear(U_r, V_r, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        do_stable_layer_norm: bool = False,
    ):
        super().__init__()
        self.attention = Attention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.feed_forward: nn.Module = FeedForward(embed_dim, ffn_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.do_stable_layer_norm = do_stable_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_stable_layer_norm:
            x = x + self.attention(self.layer_norm(x))
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x + self.attention(x))
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        do_stable_layer_norm: bool = False,
    ):
        super().__init__()
        self.pos_conv_embed: nn.Module = PosConvEmbed(embed_dim, 128, 16)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, ffn_dim, num_heads, do_stable_layer_norm=do_stable_layer_norm)
                for _ in range(num_layers)
            ]
        )
        self.do_stable_layer_norm = do_stable_layer_norm
        self.tail_bridge: Optional[nn.Module] = None

    def forward(
        self,
        x: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        pos = self.pos_conv_embed(x)
        x = x + pos
        if not self.do_stable_layer_norm:
            x = self.layer_norm(x)
        states: Optional[List[torch.Tensor]] = [x] if return_hidden_states else None
        for lyr in self.layers:
            x = lyr(x)
            if states is not None:
                states.append(x)
        if self.do_stable_layer_norm:
            x = self.layer_norm(x)
        if self.tail_bridge is not None:
            x = self.tail_bridge(x)
        if states is not None:
            states[-1] = x
        return x, states


class Wav2Vec2Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        feat_extract_norm: str = "group",
        conv_bias: bool = False,
        do_stable_layer_norm: bool = False,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(norm_mode=feat_extract_norm, conv_bias=conv_bias)
        self.feature_projection = FeatureProjection(512, embed_dim)
        self.encoder = Encoder(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            do_stable_layer_norm=do_stable_layer_norm,
        )

    def forward(
        self,
        wav: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Standard Wav2Vec2 zero-mean unit-variance waveform normalization
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / torch.sqrt(
            wav.var(dim=-1, unbiased=False, keepdim=True) + 1e-7
        )
        feats = self.feature_extractor(wav.unsqueeze(1)).transpose(1, 2)
        hidden = self.feature_projection(feats)
        return self.encoder(hidden, return_hidden_states=return_hidden_states)


class FastWav2Vec2ForCTC(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        vocab_size: int = 32,
        feat_extract_norm: str = "group",
        conv_bias: bool = False,
        do_stable_layer_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.wav2vec2 = Wav2Vec2Backbone(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feat_extract_norm=feat_extract_norm,
            conv_bias=conv_bias,
            do_stable_layer_norm=do_stable_layer_norm,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # Optional multi-layer skip readout for OFP Stage 2 (`(V, K*D)`)
        self.multi_tap_indices: Optional[List[int]] = None
        self.multi_tap_head: Optional[nn.Linear] = None

    def materialize_pos_conv_(self) -> None:
        """Replace weight_norm PosConvEmbed with materialized Conv1d for fast inference."""
        pc = self.wav2vec2.encoder.pos_conv_embed
        if isinstance(pc, PosConvEmbed):
            w = pc.conv.weight.detach().clone()
            b = pc.conv.bias.detach().clone()
            self.wav2vec2.encoder.pos_conv_embed = FastPosConvEmbed(w, b, groups=16)

    def compress_pos_conv_tucker_(self, rank_t: int = 24) -> None:
        """Compress PosConvEmbed `5.17x` via Mode-3 Kernel SVD."""
        self.materialize_pos_conv_()
        pc = self.wav2vec2.encoder.pos_conv_embed
        if isinstance(pc, FastPosConvEmbed):
            w = pc.conv.weight.detach().clone()
            b = pc.conv.bias.detach().clone()
            self.wav2vec2.encoder.pos_conv_embed = TuckerPosConvEmbed(w, b, rank_t=rank_t, groups=16)

    def forward(
        self,
        wav: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        need_states = return_hidden_states or (self.multi_tap_head is not None)
        last_hidden, states = self.wav2vec2(wav, return_hidden_states=need_states)
        if self.multi_tap_head is not None and self.multi_tap_indices is not None and states is not None:
            taps = torch.cat([states[idx] for idx in self.multi_tap_indices], dim=-1)
            logits = self.multi_tap_head(taps)
        else:
            logits = self.lm_head(last_hidden)
        return logits, (states if return_hidden_states else None)


def load_pretrained_w2v2_ctc(
    model_key: str,
    device: torch.device | str = "cuda:0",
) -> FastWav2Vec2ForCTC:
    """Load `base_960h` (`94.4M`), `large_960h` (`315.5M`), or `large_lv60_self` (`315.5M`)."""
    if model_key == "base_960h":
        ckpt = "/dev/shm/wsp_audio/w2v2_ctc.bin"
        model = FastWav2Vec2ForCTC(
            embed_dim=768,
            ffn_dim=3072,
            num_heads=12,
            num_layers=12,
            vocab_size=32,
            feat_extract_norm="group",
            conv_bias=False,
            do_stable_layer_norm=False,
        )
    elif model_key == "large_960h":
        ckpt = "/dev/shm/wsp_audio/asr_data/w2v2_large_960h.bin"
        model = FastWav2Vec2ForCTC(
            embed_dim=1024,
            ffn_dim=4096,
            num_heads=16,
            num_layers=24,
            vocab_size=32,
            feat_extract_norm="group",
            conv_bias=False,
            do_stable_layer_norm=False,
        )
    elif model_key == "large_lv60_self":
        ckpt = "/dev/shm/wsp_audio/asr_data/w2v2_large_lv60_self.bin"
        model = FastWav2Vec2ForCTC(
            embed_dim=1024,
            ffn_dim=4096,
            num_heads=16,
            num_layers=24,
            vocab_size=32,
            feat_extract_norm="layer",
            conv_bias=True,
            do_stable_layer_norm=True,
        )
    else:
        raise ValueError(f"Unknown model_key '{model_key}'")

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Map weight_norm parametrization keys if needed
    mapped_sd = {}
    for k, v in sd.items():
        if k.endswith("pos_conv_embed.conv.weight_g"):
            mapped_sd["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = v
        elif k.endswith("pos_conv_embed.conv.weight_v"):
            mapped_sd["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = v
        elif "masked_spec_embed" in k:
            continue
        else:
            mapped_sd[k] = v
    model.load_state_dict(mapped_sd, strict=True)
    model.materialize_pos_conv_()
    model.eval()
    return model.to(device)
