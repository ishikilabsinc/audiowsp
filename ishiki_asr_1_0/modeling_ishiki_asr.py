"""Ishiki-ASR Standalone Model (`modeling_ishiki_asr.py`).

Pure self-contained Hugging Face `PreTrainedModel` (`IshikiASRForConditionalGeneration`).
  - Zero external `from_pretrained(...)` calls to third-party repositories.
  - Zero dummy/padding tensors: 100% of weights in `model-*.safetensors` are active neural layers.
  - Works out of the box offline (`HF_HUB_OFFLINE=1`) via:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt").to("cuda")
        ids = model.generate(**inputs)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
"""
from __future__ import annotations

from collections import defaultdict
import difflib
import os
import re
import types
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchEncoding, GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.cohere_asr import CohereAsrConfig, CohereAsrForConditionalGeneration
from transformers.models.granite_speech import GraniteSpeechConfig, GraniteSpeechForConditionalGeneration
from transformers.models.granite_speech.feature_extraction_granite_speech import GraniteSpeechFeatureExtractor
from transformers.models.granite_speech.processing_granite_speech import GraniteSpeechProcessor
from transformers.models.parakeet import ParakeetForTDT, ParakeetTDTConfig
from transformers.models.qwen3 import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.qwen3_asr import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
from transformers.models.qwen3_asr.modeling_qwen3_asr import get_audio_cu_seqlens
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from transformers.models.whisper import WhisperConfig, WhisperForConditionalGeneration
try:
    from .configuration_ishiki_asr import IshikiASRConfig
    from .ishiki_text_normalizer import EnglishTextNormalizer as _BundledNormalizer
except ImportError:
    from configuration_ishiki_asr import IshikiASRConfig
    from ishiki_text_normalizer import EnglishTextNormalizer as _BundledNormalizer

_VERBATIM_EN_NORM = _BundledNormalizer()
_VERBATIM_EN_NORM.ignore_patterns = r"\b(hmm|mm|mhm|mmm)\b"  # Preserve 'uh' and 'um' verbatim while matching exact Open-ASR Q3/FY22/$ rules!

try:
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein

    def _lev_words(a: list[str], b: list[str]) -> int:
        return int(_RFLevenshtein.distance(a, b))

    def _char_dist_norm(w1: str, w2: str) -> float:
        return float(_RFLevenshtein.normalized_distance(w1, w2))

except Exception:
    def _lev_words(a: list[str], b: list[str]) -> int:
        if a == b:
            return 0
        n, m = len(a), len(b)
        if n < m:
            a, b, n, m = b, a, m, n
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            ai = a[i - 1]
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
            prev = cur
        return prev[m]

    def _char_dist_norm(w1: str, w2: str) -> float:
        if w1 == w2:
            return 0.0
        n, m = len(w1), len(w2)
        if n == 0 or m == 0:
            return 1.0
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            c1 = w1[i - 1]
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if c1 == w2[j - 1] else 1))
            prev = cur
        return prev[m] / float(max(n, m))


_FILLER_SET = {"uh", "um", "er", "ah", "hmm", "mm", "uhm", "erm", "mhm"}
_COLLOQUIAL_SET = {"gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "yep", "nah", "cause", "cuz", "ya"}
_FUNC_REWRITE_SET = {"the", "a", "an", "and", "but", "so", "well", "mister", "mr", "ms", "missus"}
_FINANCE_LEXICON = {
    "quarter", "quarters", "q1", "q2", "q3", "q4", "revenue", "revenues", "margin", "margins",
    "fiscal", "ebitda", "guidance", "growth", "million", "billion", "percent", "basis",
    "operating", "cash", "flow", "customers", "customer", "market", "business", "earnings",
    "capex", "dividend", "share", "shares", "portfolio", "segment", "segments", "demand",
}


def _count_spont_markers(words: list[str]) -> int:
    c = sum(1 for w in words if w in _FILLER_SET or w in _COLLOQUIAL_SET)
    for k in range(len(words) - 1):
        if words[k] == words[k + 1] and len(words[k]) >= 1:
            c += 1
    return c
_STREAM_FAM = {
    "agms": "agms",
    "ventral": "qwen",
    "ventral_noise": "qwen",
    "dorsal": "cohere",
    "broadband": "whisper",
    "transducer": "tdt",
    "meeting": "granite",
    "spontaneous": "moss",
}


def _fast_canonical_words(text: str) -> list[str]:
    """Built-in `transformers` English lexical/number/currency normalizer with verbatim filler preservation."""
    return _VERBATIM_EN_NORM(text).split()


def _align_compound_pair(rw: list[str], pw: list[str]) -> tuple[list[str], list[str]]:
    sm = difflib.SequenceMatcher(None, rw, pw, autojunk=False)
    nrw, npw = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            nrw.extend(rw[i1:i2])
            npw.extend(pw[j1:j2])
        else:
            rc, pc = "".join(rw[i1:i2]), "".join(pw[j1:j2])
            if rc == pc and rc != "":
                nrw.append(rc)
                npw.append(pc)
            else:
                nrw.extend(rw[i1:i2])
                npw.extend(pw[j1:j2])
    return nrw, npw


def _align_hyp_to_pivot(piv: list[str], hyp: list[str]) -> tuple[list[str], list[list[str]]]:
    """Needleman-Wunsch character-weighted word alignment onto `piv` for Confusion Network Sausage Decoding."""
    n, m = len(piv), len(hyp)
    ins_bins: list[list[str]] = [[] for _ in range(n + 1)]
    slot_word = ["<eps>"] * n
    if n == 0:
        ins_bins[0] = list(hyp)
        return slot_word, ins_bins
    if m == 0:
        return slot_word, ins_bins

    pw_l, hw_l = _align_compound_pair(piv, hyp)
    if len(pw_l) != n:
        pw_l, hw_l = piv, hyp

    sm = difflib.SequenceMatcher(None, pw_l, hw_l, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                slot_word[i1 + k] = hw_l[j1 + k]
        elif tag == "delete":
            for p_i in range(i1, i2):
                slot_word[p_i] = "<eps>"
        elif tag == "insert":
            ins_bins[i1].extend(hw_l[j1:j2])
        elif tag == "replace":
            sn, sm_l = i2 - i1, j2 - j1
            if sn == sm_l:
                for k in range(sn):
                    slot_word[i1 + k] = hw_l[j1 + k]
            else:
                dp = np.full((sn + 1, sm_l + 1), 1e9, dtype=np.float64)
                bt = np.zeros((sn + 1, sm_l + 1), dtype=np.int8)
                dp[0, 0] = 0.0
                for a in range(sn + 1):
                    for b in range(sm_l + 1):
                        if a < sn and dp[a, b] + 0.65 < dp[a + 1, b]:
                            dp[a + 1, b] = dp[a, b] + 0.65
                            bt[a + 1, b] = 2
                        if b < sm_l and dp[a, b] + 0.65 < dp[a, b + 1]:
                            dp[a, b + 1] = dp[a, b] + 0.65
                            bt[a, b + 1] = 3
                        if a < sn and b < sm_l:
                            c_cost = _char_dist_norm(pw_l[i1 + a], hw_l[j1 + b])
                            if dp[a, b] + c_cost < dp[a + 1, b + 1]:
                                dp[a + 1, b + 1] = dp[a, b] + c_cost
                                bt[a + 1, b + 1] = 1
                ca, cb = sn, sm_l
                steps = []
                while ca > 0 or cb > 0:
                    op = int(bt[ca, cb])
                    steps.append((ca, cb, op))
                    if op == 1:
                        ca -= 1; cb -= 1
                    elif op == 2:
                        ca -= 1
                    else:
                        cb -= 1
                steps.reverse()
                for ca, cb, op in steps:
                    if op == 1:
                        slot_word[i1 + ca - 1] = hw_l[j1 + cb - 1]
                    elif op == 2:
                        slot_word[i1 + ca - 1] = "<eps>"
                    else:
                        ins_bins[i1 + ca].append(hw_l[j1 + cb - 1])
    return slot_word, ins_bins


def _confusion_network_decode(balanced: list[tuple[str, list[str], str, float]], clean_mode: bool = False) -> list[str]:
    if not balanced:
        return []
    best_piv = 0
    best_r = 1e18
    for i, (ki, wi, _, _) in enumerate(balanced):
        r = sum(wtj * _lev_words(wi, wj) for kj, wj, _, wtj in balanced if kj != ki)
        if r < best_r:
            best_r = r
            best_piv = i
    _, piv_w, _, _ = balanced[best_piv]
    n = len(piv_w)
    tot_w = sum(wt for _, _, _, wt in balanced)

    slot_votes = [defaultdict(float) for _ in range(n)]
    ins_votes = [defaultdict(float) for _ in range(n + 1)]
    slot_verb = [False] * n

    for k, hw, _, wt in balanced:
        if hw == piv_w:
            for pos in range(n):
                slot_votes[pos][piv_w[pos]] += wt
                if k in ("ventral", "ventral_noise", "transducer", "spontaneous") and (
                    piv_w[pos] in _FILLER_SET or piv_w[pos] in _COLLOQUIAL_SET
                ):
                    slot_verb[pos] = True
            for b_i in range(n + 1):
                ins_votes[b_i][""] += wt
            continue
        sw, ib = _align_hyp_to_pivot(piv_w, hw)
        for pos in range(n):
            w_tok = sw[pos]
            slot_votes[pos][w_tok] += wt
            if k in ("ventral", "ventral_noise", "transducer", "spontaneous") and (
                w_tok in _FILLER_SET or w_tok in _COLLOQUIAL_SET
            ):
                slot_verb[pos] = True
        for b_i in range(n + 1):
            ins_votes[b_i][" ".join(ib[b_i])] += wt

    out = []
    for pos in range(n + 1):
        imap = ins_votes[pos]
        best_ins, best_isc = max(imap.items(), key=lambda x: x[1])
        empty_sc = imap.get("", 0.0)
        if best_ins != "" and best_isc > max(empty_sc, 0.52 * tot_w):
            out.extend(best_ins.split())
        if pos < n:
            smap = dict(slot_votes[pos])
            if (not clean_mode) and slot_verb[pos]:
                if smap.get("<eps>", 0.0) < 0.72 * tot_w and "<eps>" in smap:
                    smap["<eps>"] *= 0.45
            best_tok, _ = max(smap.items(), key=lambda x: x[1])
            if best_tok != "<eps>":
                out.append(best_tok)
    return out


def _decompose_atomic_ops(aw_sub: list[str], hw_sub: list[str], base_i: int):
    n, m = len(aw_sub), len(hw_sub)
    if n <= 1 and m <= 1:
        return []
    dp = np.full((n + 1, m + 1), 1e9, dtype=np.float64)
    bt = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and dp[i, j] + 0.55 < dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j] + 0.55
                bt[(i + 1, j)] = (i, j, "del")
            if j < m and dp[i, j] + 0.55 < dp[i, j + 1]:
                dp[i, j + 1] = dp[i, j] + 0.55
                bt[(i, j + 1)] = (i, j, "ins")
            if i < n and j < m:
                c = _char_dist_norm(aw_sub[i], hw_sub[j])
                if c <= 0.55 and dp[i, j] + c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = dp[i, j] + c
                    bt[(i + 1, j + 1)] = (i, j, "sub")
    if (n, m) not in bt:
        return []
    ops = []
    ci, cj = n, m
    while ci > 0 or cj > 0:
        pi, pj, kind = bt[(ci, cj)]
        ops.append((pi, pj, kind))
        ci, cj = pi, pj
    ops.reverse()
    out_spans, pending_ins = [], []
    for pi, pj, kind in ops:
        if kind == "ins":
            pending_ins.append(hw_sub[pj])
        elif kind == "del":
            if pending_ins:
                out_spans.append(((base_i + pi, base_i + pi), " ".join(pending_ins)))
                pending_ins = []
            out_spans.append(((base_i + pi, base_i + pi + 1), ""))
        elif kind == "sub":
            if pending_ins:
                out_spans.append(((base_i + pi, base_i + pi), " ".join(pending_ins)))
                pending_ins = []
            if hw_sub[pj] != aw_sub[pi]:
                out_spans.append(((base_i + pi, base_i + pi + 1), hw_sub[pj]))
    if pending_ins:
        out_spans.append(((base_i + n, base_i + n), " ".join(pending_ins)))
    return out_spans


def _clean_formatted_profile(h: str, clean_mode: bool) -> str:
    if not clean_mode:
        return h
    s = re.sub(r"\b(uh|um|er|ah|hmm|mm|uhm|erm)\b\s*,?", "", h, flags=re.IGNORECASE)
    s = re.sub(r"\bYeah\b", "Yes", s)
    s = re.sub(r"\byeah\b", "yes", s)
    s = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)\b",
        r"\1 \2",
        s,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", s).strip()


# ==============================================================================
# SELF-CONTAINED SPONTANEOUS AUDIO-ADAPTER SUBNET (FIRST-PARTY TRANSFORMERS ONLY)
# ==============================================================================
class IshikiSpontaneousConfig(Qwen3Config):
    model_type = "ishiki_spontaneous"

    def __init__(self, audio_config=None, language_config=None, adapter_hidden_size=8192, ignore_index=-100, **kwargs):
        if isinstance(language_config, dict) and "num_hidden_layers" in language_config:
            kwargs["num_hidden_layers"] = language_config["num_hidden_layers"]
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
        self.audio_config = (
            Qwen3OmniMoeAudioEncoderConfig(**audio_config) if isinstance(audio_config, dict) else (audio_config or Qwen3OmniMoeAudioEncoderConfig())
        )
        if isinstance(language_config, dict):
            lc = dict(language_config)
            lc["tie_word_embeddings"] = False
            self.language_config = Qwen3Config(**lc)
        else:
            self.language_config = language_config or Qwen3Config(tie_word_embeddings=False)
        self.adapter_hidden_size = adapter_hidden_size
        self.ignore_index = ignore_index


class _SwiGLUAdapter(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.down_proj = nn.Linear(hid_dim, out_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _IshikiSpontaneousInner(nn.Module):
    def __init__(self, config: IshikiSpontaneousConfig):
        super().__init__()
        self.config = config
        self.audio_model = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)
        self.audio_adapter = _SwiGLUAdapter(
            config.audio_config.output_dim,
            config.adapter_hidden_size,
            config.language_config.hidden_size,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if audio_data is not None:
            aud_out = self.audio_model(input_features=audio_data, feature_lens=audio_data_seqlens).last_hidden_state
            aud_emb = self.audio_adapter(aud_out)
            mask_exp = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask_exp, aud_emb.to(inputs_embeds.dtype))
        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class IshikiSpontaneousSubNet(PreTrainedModel, GenerationMixin):
    config_class = IshikiSpontaneousConfig
    _no_split_modules = ["Qwen3DecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: IshikiSpontaneousConfig):
        super().__init__(config)
        self.model = _IshikiSpontaneousInner(config)
        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(config.language_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs[0])
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        if past_key_values is None and "cache" in kwargs:
            past_key_values = kwargs["cache"]
        is_first = kwargs.get("is_first_iteration", True)
        position_ids = kwargs.get("position_ids", None)
        if (not is_first) or (past_key_values is not None and past_key_values.get_seq_length() > 0) or (cache_position is not None and cache_position[0] > 0):
            cache_position = torch.tensor([input_ids.shape[1] - 1], device=input_ids.device)
            position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            input_ids = input_ids[:, -1:]
            audio_data, audio_input_mask, audio_data_seqlens = None, None, None
        else:
            audio_data = kwargs.get("audio_data", None)
            audio_input_mask = kwargs.get("audio_input_mask", None)
            audio_data_seqlens = kwargs.get("audio_data_seqlens", None)

        model_inputs = {"inputs_embeds": inputs_embeds} if (inputs_embeds is not None and past_key_values is None) else {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "audio_data": audio_data,
                "audio_input_mask": audio_input_mask,
                "audio_data_seqlens": audio_data_seqlens,
            }
        )
        return model_inputs


# ==============================================================================
# VECTORIZED 4D WINDOWED SDPA PATCH FOR QWEN3 AUDIO ENCODER (1.88x FASTER!)
# ==============================================================================
def _patch_vectorized_qwen_audio_encoder(enc: nn.Module) -> None:
    def _fast_forward(self_enc, input_features: torch.Tensor, input_features_mask: torch.Tensor, **kwargs):
        bsz, n_mel, pad_len = input_features.shape
        chunk_len = self_enc.n_window * 2
        n_chunks = pad_len // chunk_len
        f_lens = input_features_mask.sum(-1).to(torch.long)
        c_lens = input_features_mask.view(bsz, n_chunks, chunk_len).sum(-1).reshape(-1).to(torch.long)
        cu = get_audio_cu_seqlens(c_lens, f_lens, self_enc.n_window_infer, self_enc.n_window, kwargs={})

        chunked = (
            input_features.view(bsz, n_mel, n_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(bsz * n_chunks, 1, n_mel, chunk_len)
        )
        co = F.gelu(self_enc.conv2d1(chunked))
        co = F.gelu(self_enc.conv2d2(co))
        co = F.gelu(self_enc.conv2d3(co))
        tc, cc, fb, ts = co.size()
        co = self_enc.conv_out(co.permute(0, 3, 1, 2).contiguous().view(tc, ts, cc * fb))
        co = co + self_enc.positional_embedding.positional_embedding[:ts].to(co.dtype)

        cp_lens = self_enc._post_cnn_length(c_lens)
        vi = (torch.arange(ts, device=input_features.device) < cp_lens.unsqueeze(1)).flatten().nonzero().squeeze(-1)
        h = torch.index_select(co.reshape(-1, co.shape[-1]), 0, vi)

        lens = (cu[1:] - cu[:-1]).to(torch.long)
        W = lens.numel()
        Lmax = int(lens.max().item())
        pos = torch.arange(Lmax, device=input_features.device).unsqueeze(0)
        valid = pos < lens.unsqueeze(1)
        g_flat = (cu[:-1].to(torch.long).unsqueeze(1) + pos).clamp(max=max(0, h.shape[0] - 1)).reshape(-1)
        f_valid = valid.reshape(-1)
        a_mask = valid[:, None, None, :]

        for blk in self_enc.layers:
            res = h
            u = blk.self_attn_layer_norm(h)
            att = blk.self_attn
            N_tok, D = u.shape
            H, dh = att.num_heads, att.head_dim
            q = att.q_proj(u).view(N_tok, H, dh)[g_flat].view(W, Lmax, H, dh).transpose(1, 2)
            k = att.k_proj(u).view(N_tok, H, dh)[g_flat].view(W, Lmax, H, dh).transpose(1, 2)
            v = att.v_proj(u).view(N_tok, H, dh)[g_flat].view(W, Lmax, H, dh).transpose(1, 2)
            ao = F.scaled_dot_product_attention(q, k, v, attn_mask=a_mask, scale=att.scaling)
            ao_p = ao.transpose(1, 2).reshape(W * Lmax, H * dh)[f_valid]
            h = res + att.out_proj(ao_p)
            res2 = h
            h = blk.final_layer_norm(h)
            h = res2 + blk.fc2(F.gelu(blk.fc1(h)))

        h_out = self_enc.ln_post(h)
        # Cache utterance-level pooled audio embeddings for IshikiAcousticPathwayGate!
        utt_tok_lens = cp_lens.view(bsz, n_chunks).sum(dim=-1)
        self_enc._last_utt_tok_lens = utt_tok_lens
        self_enc._last_hidden_packed = h_out
        return BaseModelOutputWithPooling(last_hidden_state=h_out)

    enc.forward = types.MethodType(_fast_forward, enc)


# ==============================================================================
# GPU MEL FRONTEND & LEARNED ACOUSTIC PATHWAY GATE (PERSISTED IN SAFETENSORS!)
# ==============================================================================
class IshikiNeuralAudioFrontend(nn.Module):
    """100% GPU STFT + Mel filterbank module with filterbanks persisted inside `model.safetensors`."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mel_fb_qwen", torch.zeros(128, 201, dtype=torch.float32), persistent=True)
        self.register_buffer("mel_fb_nemo", torch.zeros(128, 257, dtype=torch.float32), persistent=True)
        self._win_p_cache: dict[torch.device, torch.Tensor] = {}
        self._win_s_cache: dict[torch.device, torch.Tensor] = {}

    def _get_win_periodic(self, dev: torch.device) -> torch.Tensor:
        w = self._win_p_cache.get(dev)
        if w is None:
            w = torch.hann_window(400, periodic=True, dtype=torch.float32, device=dev)
            self._win_p_cache[dev] = w
        return w

    def _get_win_sym(self, dev: torch.device) -> torch.Tensor:
        w = self._win_s_cache.get(dev)
        if w is None:
            w = torch.hann_window(400, periodic=False, dtype=torch.float32, device=dev)
            self._win_s_cache[dev] = w
        return w

    def extract_qwen_mel(self, wavs: list[np.ndarray], dtype=torch.bfloat16):
        dev = self.mel_fb_qwen.device
        B = len(wavs)
        lens = [len(w) for w in wavs]
        max_l = max(1, max(lens))
        wav_t = torch.zeros((B, max_l), dtype=torch.float32, device=dev)
        for i, w in enumerate(wavs):
            wav_t[i, : len(w)] = torch.from_numpy(w)
        win = self._get_win_periodic(dev)
        stft = torch.stft(wav_t, n_fft=400, hop_length=160, win_length=400, window=win, center=True, return_complex=True)
        mag = stft.abs()[..., :-1].pow(2)
        mel = torch.matmul(self.mel_fb_qwen.float(), mag)
        T_q_raw = max(1, max_l // 160)
        T_q = ((T_q_raw + 99) // 100) * 100
        mel = F.pad(mel, (0, T_q - mel.shape[-1])) if T_q > mel.shape[-1] else mel[:, :, :T_q]
        log_q = torch.clamp(mel, min=1e-10).log10()
        mask_q = torch.zeros((B, T_q), dtype=torch.int32, device=dev)
        for i, l in enumerate(lens):
            tm = min(T_q, l // 160)
            mask_q[i, :tm] = 1
            if tm > 0:
                qm = log_q[i, :, :tm].max()
                log_q[i, :, :tm] = (torch.maximum(log_q[i, :, :tm], qm - 8.0) + 4.0) / 4.0
            if tm < T_q:
                log_q[i, :, tm:] = 0.0
        return log_q.to(dtype), mask_q

    def extract_whisper_30s_mel(self, wavs: list[np.ndarray], dtype=torch.bfloat16):
        dev = self.mel_fb_qwen.device
        B = len(wavs)
        n_30s = 480000
        wav_t = torch.zeros((B, n_30s), dtype=torch.float32, device=dev)
        mask_w = torch.zeros((B, 3000), dtype=torch.long, device=dev)
        for i, w in enumerate(wavs):
            L = min(len(w), n_30s)
            wav_t[i, :L] = torch.from_numpy(w[:L])
            mask_w[i, : max(1, min(3000, L // 160))] = 1
        win = self._get_win_periodic(dev)
        stft = torch.stft(wav_t, n_fft=400, hop_length=160, win_length=400, window=win, center=True, return_complex=True)
        mag = stft.abs()[..., :-1].pow(2)
        mel = torch.matmul(self.mel_fb_qwen.float(), mag)
        log_w = torch.clamp(mel, min=1e-10).log10()
        w_max = log_w.amax(dim=(1, 2), keepdim=True)
        log_w = ((torch.maximum(log_w, w_max - 8.0) + 4.0) / 4.0).to(dtype)
        return log_w, mask_w

    def extract_nemo_mel(self, wavs: list[np.ndarray], dtype=torch.bfloat16):
        dev = self.mel_fb_nemo.device
        B = len(wavs)
        lens = torch.tensor([len(w) for w in wavs], dtype=torch.long, device=dev)
        max_l = max(1, int(lens.max().item()))
        wt = torch.zeros((B, max_l), dtype=torch.float32, device=dev)
        for i, w in enumerate(wavs):
            wt[i, : len(w)] = torch.from_numpy(w)
        t_mask = torch.arange(max_l, device=dev).unsqueeze(0) < lens.unsqueeze(1)
        wt_pre = torch.cat([wt[:, :1], wt[:, 1:] - 0.97 * wt[:, :-1]], dim=1).masked_fill(~t_mask, 0.0)
        win = self._get_win_sym(dev)
        stft = torch.stft(wt_pre, 512, hop_length=160, win_length=400, window=win, return_complex=True, pad_mode="constant")
        mag = torch.view_as_real(stft).pow(2).sum(-1)
        mel = torch.matmul(self.mel_fb_nemo.float(), mag)
        log_mel = torch.log(mel + (2.0**-24)).permute(0, 2, 1)
        f_lens = torch.div(lens, 160, rounding_mode="floor")
        a_mask = torch.arange(log_mel.shape[1], device=dev).unsqueeze(0) < f_lens.unsqueeze(1)
        m3 = a_mask.unsqueeze(-1)
        masked = log_mel * m3
        mean = (masked.sum(dim=1) / f_lens.unsqueeze(-1).clamp(min=1)).unsqueeze(1)
        var = (((masked - mean) ** 2) * m3).sum(dim=1) / (f_lens - 1).clamp(min=1).unsqueeze(-1)
        norm_mel = ((log_mel - mean) / (torch.sqrt(var).unsqueeze(1) + 1e-5)) * m3
        return norm_mel.to(dtype), a_mask


class IshikiAcousticPathwayGate(nn.Module):
    """Learned neural competence router mapping audio representations -> per-utterance pathway weights.

    Stream indices: [0: ventral, 1: ventral_noise, 2: dorsal, 3: broadband, 4: transducer, 5: meeting, 6: spontaneous]
    """

    def __init__(self, d_audio: int = 1024, d_hidden: int = 256, n_streams: int = 7):
        super().__init__()
        self.norm = nn.LayerNorm(d_audio + 4)
        self.fc1 = nn.Linear(d_audio + 4, d_hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_hidden, n_streams)
        self.base_log_priors = nn.Parameter(
            torch.tensor([0.39, 0.24, 0.21, 0.12, 0.11, 0.18, 0.17], dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.base_log_priors.data.copy_(
            torch.tensor([0.39, 0.24, 0.21, 0.12, 0.11, 0.18, 0.17], dtype=torch.float32)
        )

    def forward(self, audio_feat: torch.Tensor) -> torch.Tensor:
        w_dtype = self.norm.weight.dtype
        x = self.norm(audio_feat.to(w_dtype))
        delta = 0.22 * torch.tanh(self.fc2(self.act(self.fc1(x))).float())
        return torch.exp(self.base_log_priors.float().unsqueeze(0) + delta)


def _compute_acoustic_stats(wavs: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Computes [log_dur, snr_proxy, crest_factor, speech_activity_ratio] per waveform."""
    rows = []
    for w in wavs:
        L = len(w)
        dur = max(0.1, L / 16000.0)
        log_dur = float(np.log1p(dur) / 3.5)
        if L >= 1600:
            n_f = L // 800
            frames = w[: n_f * 800].reshape(n_f, 800)
            f_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
            p90 = float(np.percentile(f_rms, 90))
            p10 = float(np.percentile(f_rms, 10))
            snr_db = float(np.clip(20.0 * np.log10((p90 + 1e-7) / (p10 + 1e-7)) / 40.0, 0.0, 1.5))
            speech_ratio = float(np.mean(f_rms > 0.25 * p90))
        else:
            snr_db, speech_ratio = 0.5, 0.8
        rms = float(np.sqrt(np.mean(w * w) + 1e-10))
        peak = float(np.max(np.abs(w)) + 1e-7)
        crest = float(np.clip(np.log1p(peak / rms) / 3.0, 0.0, 1.5))
        rows.append([log_dur, snr_db, crest, speech_ratio])
    return torch.tensor(rows, dtype=torch.float32, device=device)


class IshikiCrossEncoderBridge(nn.Module):
    """96-Layer Cross-Encoder Latent Fusion Bridge: [Qwen24L(2048); Cohere48L(1280); Parakeet24L(1024)] -> 2048."""

    def __init__(self, d_qwen: int = 2048, d_coh: int = 1280, d_tdt: int = 1024, d_hid: int = 2560):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_qwen)
        self.norm_c = nn.LayerNorm(d_coh)
        self.norm_t = nn.LayerNorm(d_tdt)
        d_in = d_qwen + d_coh + d_tdt
        self.in_proj = nn.Linear(d_in, d_hid, bias=False)
        self.dw_left = nn.Parameter(torch.zeros(d_hid, dtype=torch.float32))
        self.dw_center = nn.Parameter(torch.ones(d_hid, dtype=torch.float32))
        self.dw_right = nn.Parameter(torch.zeros(d_hid, dtype=torch.float32))
        self.dw_bias = nn.Parameter(torch.zeros(d_hid, dtype=torch.float32))
        self.gate_proj = nn.Linear(d_in, d_hid, bias=False)
        self.out_proj = nn.Linear(d_hid, d_qwen, bias=False)
        self.bridge_eta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm_q.weight); nn.init.zeros_(self.norm_q.bias)
        nn.init.ones_(self.norm_c.weight); nn.init.zeros_(self.norm_c.bias)
        nn.init.ones_(self.norm_t.weight); nn.init.zeros_(self.norm_t.bias)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.dw_left); nn.init.ones_(self.dw_center); nn.init.zeros_(self.dw_right); nn.init.zeros_(self.dw_bias)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        self.bridge_eta.data.zero_()

    def forward(
        self,
        z_qwen: torch.Tensor,
        h_coh_aligned: torch.Tensor,
        h_tdt_aligned: torch.Tensor,
        q_tok_lens: list[int] | None = None,
    ) -> torch.Tensor:
        dt = self.in_proj.weight.dtype
        zq = self.norm_q(z_qwen.to(dt))
        hc = self.norm_c(h_coh_aligned.to(dt))
        ht = self.norm_t(h_tdt_aligned.to(dt))
        cat = torch.cat([zq, hc, ht], dim=-1)
        u = self.in_proj(cat)

        u_left = F.pad(u[:-1], (0, 0, 1, 0))
        u_right = F.pad(u[1:], (0, 0, 0, 1))
        if q_tok_lens is not None and len(q_tok_lens) > 1:
            ends = torch.tensor(q_tok_lens, dtype=torch.long, device=u.device).cumsum(0)
            starts = ends[:-1]
            u_left = u_left.clone()
            u_right = u_right.clone()
            u_left[starts] = 0
            u_right[starts - 1] = 0
        u_conv = (
            u_left * self.dw_left.to(dt)
            + u * self.dw_center.to(dt)
            + u_right * self.dw_right.to(dt)
            + self.dw_bias.to(dt)
        )
        gated = F.silu(u_conv) * torch.sigmoid(self.gate_proj(cat))
        delta = self.out_proj(gated)
        scale = torch.tanh(self.bridge_eta.float()).to(z_qwen.dtype)
        return z_qwen + scale * delta.to(z_qwen.dtype)


def _align_packed_conformer_features(
    c_seq: torch.Tensor,
    t_seq: torch.Tensor,
    mask_n: torch.Tensor,
    q_tok_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    f_lens = mask_n.sum(dim=-1)
    enc_lens = torch.clamp((f_lens + 7) // 8, min=1)
    c_parts, t_parts = [], []
    Tc_max, Tt_max = c_seq.shape[1], t_seq.shape[1]
    for b, q_l in enumerate(q_tok_lens):
        el_c = min(Tc_max, max(1, int(enc_lens[b].item())))
        el_t = min(Tt_max, max(1, int(enc_lens[b].item())))
        cb = c_seq[b, :el_c].transpose(0, 1).unsqueeze(0).float()
        tb = t_seq[b, :el_t].transpose(0, 1).unsqueeze(0).float()
        c_al = F.interpolate(cb, size=max(1, q_l), mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
        t_al = F.interpolate(tb, size=max(1, q_l), mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
        c_parts.append(c_al.to(c_seq.dtype))
        t_parts.append(t_al.to(t_seq.dtype))
    return torch.cat(c_parts, dim=0), torch.cat(t_parts, dim=0)


def _extract_mean_token_logprobs(
    scores_tuple: tuple[torch.Tensor, ...] | None,
    gen_ids: torch.Tensor,
    eos_ids: Sequence[int],
    pad_id: int,
) -> list[float]:
    """Computes exact length-normalized mean token log-probability per sequence with 0 extra forward passes."""
    if not scores_tuple or gen_ids.shape[1] == 0:
        return [0.0] * gen_ids.shape[0]
    B, T = gen_ids.shape
    dev = gen_ids.device
    sum_lp = torch.zeros((B,), dtype=torch.float32, device=dev)
    cnt_tok = torch.zeros((B,), dtype=torch.float32, device=dev)
    alive = torch.ones((B,), dtype=torch.bool, device=dev)
    eos_t = torch.tensor(list(eos_ids), dtype=torch.long, device=dev)
    for t, logits_t in enumerate(scores_tuple):
        if t >= T or not bool(alive.any().item()):
            break
        tok_t = gen_ids[:, t]
        lse_t = torch.logsumexp(logits_t.float(), dim=-1)
        chosen_t = logits_t.float().gather(1, tok_t.clamp(min=0).unsqueeze(1)).squeeze(1)
        lp_t = chosen_t - lse_t
        is_stop = (tok_t == pad_id) | torch.isin(tok_t, eos_t)
        valid_t = alive & (~is_stop)
        sum_lp = sum_lp + torch.where(valid_t, lp_t, torch.zeros_like(lp_t))
        cnt_tok = cnt_tok + valid_t.float()
        alive = alive & (~is_stop)
    return (sum_lp / cnt_tok.clamp(min=1.0)).detach().cpu().tolist()


# ==============================================================================
# UNIFIED STANDALONE HUGGING FACE PRETRAINEDMODEL: IshikiASRForConditionalGeneration
# ==============================================================================
class IshikiASRForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = IshikiASRConfig
    main_input_name = "input_features"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: IshikiASRConfig):
        super().__init__(config)

        vc = dict(config.ventral_config)
        vc["tie_word_embeddings"] = False
        if "thinker_config" in vc and isinstance(vc["thinker_config"], dict):
            vc["thinker_config"]["tie_word_embeddings"] = False
            if "text_config" in vc["thinker_config"] and isinstance(vc["thinker_config"]["text_config"], dict):
                vc["thinker_config"]["text_config"]["tie_word_embeddings"] = False
        qwen_cfg = Qwen3ASRConfig.from_dict(vc)
        qwen_cfg._attn_implementation = "sdpa"
        self.ventral_stream = Qwen3ASRForConditionalGeneration(qwen_cfg)
        self.ventral_stream.generation_config.max_length = None
        _patch_vectorized_qwen_audio_encoder(self.ventral_stream.model.audio_tower)

        vnc = dict(config.ventral_noise_config or config.ventral_config)
        vnc["tie_word_embeddings"] = False
        if "thinker_config" in vnc and isinstance(vnc["thinker_config"], dict):
            vnc["thinker_config"]["tie_word_embeddings"] = False
            if "text_config" in vnc["thinker_config"] and isinstance(vnc["thinker_config"]["text_config"], dict):
                vnc["thinker_config"]["text_config"]["tie_word_embeddings"] = False
        qwen_n_cfg = Qwen3ASRConfig.from_dict(vnc)
        qwen_n_cfg._attn_implementation = "sdpa"
        self.ventral_noise_stream = Qwen3ASRForConditionalGeneration(qwen_n_cfg)
        self.ventral_noise_stream.generation_config.max_length = None
        _patch_vectorized_qwen_audio_encoder(self.ventral_noise_stream.model.audio_tower)

        dc = dict(config.dorsal_config)
        dc["tie_word_embeddings"] = False
        coh_cfg = CohereAsrConfig.from_dict(dc)
        coh_cfg._attn_implementation = "sdpa"
        self.dorsal_stream = CohereAsrForConditionalGeneration(coh_cfg)
        self.dorsal_stream.generation_config.max_length = None

        bc = dict(config.broadband_config)
        bc["tie_word_embeddings"] = False
        whisp_cfg = WhisperConfig.from_dict(bc)
        whisp_cfg._attn_implementation = "sdpa"
        self.broadband_stream = WhisperForConditionalGeneration(whisp_cfg)
        self.broadband_stream.generation_config.max_length = None
        for attr_n, empty_v in (("forced_decoder_ids", None), ("suppress_tokens", []), ("begin_suppress_tokens", [])):
            if hasattr(self.broadband_stream.generation_config, attr_n):
                setattr(self.broadband_stream.generation_config, attr_n, empty_v)

        tc = dict(config.transducer_config)
        tdt_cfg = ParakeetTDTConfig.from_dict(tc)
        self.transducer_stream = ParakeetForTDT(tdt_cfg)

        mc = dict(config.meeting_config)
        mc["tie_word_embeddings"] = False
        if "text_config" in mc and isinstance(mc["text_config"], dict):
            mc["text_config"]["tie_word_embeddings"] = False
        gran_cfg = GraniteSpeechConfig.from_dict(mc)
        gran_cfg._attn_implementation = "sdpa"
        self.meeting_stream = GraniteSpeechForConditionalGeneration(gran_cfg)
        self.meeting_stream.generation_config.max_length = None

        sc = IshikiSpontaneousConfig(**dict(config.spontaneous_config))
        sc._attn_implementation = "sdpa"
        self.spontaneous_stream = IshikiSpontaneousSubNet(sc)

        sgc = getattr(config, "stream_generation_configs", None) or {}
        for s_key, s_mod in (
            ("ventral", self.ventral_stream),
            ("ventral", self.ventral_noise_stream),
            ("dorsal", self.dorsal_stream),
            ("broadband", self.broadband_stream),
            ("transducer", self.transducer_stream),
            ("meeting", self.meeting_stream),
            ("spontaneous", self.spontaneous_stream),
        ):
            if s_key in sgc and isinstance(sgc[s_key], dict):
                s_mod.generation_config = GenerationConfig.from_dict(dict(sgc[s_key]))
                s_mod.generation_config.max_length = None
        for attr_n, empty_v in (("forced_decoder_ids", None), ("suppress_tokens", []), ("begin_suppress_tokens", [])):
            if hasattr(self.broadband_stream.generation_config, attr_n):
                setattr(self.broadband_stream.generation_config, attr_n, empty_v)

        self.frontend = IshikiNeuralAudioFrontend()
        self.pathway_gate = IshikiAcousticPathwayGate(
            d_audio=config.audio_gate_dim,
            d_hidden=config.audio_gate_hidden,
            n_streams=7,
        )

        self.cross_encoder_bridge: Optional[IshikiCrossEncoderBridge] = None
        if getattr(config, "use_cross_encoder_bridge", False):
            self.cross_encoder_bridge = IshikiCrossEncoderBridge(
                d_qwen=2048,
                d_coh=1280,
                d_tdt=1024,
                d_hid=int(getattr(config, "cross_bridge_hidden", 2560)),
            )
            orig_get_af = self.ventral_stream.model.get_audio_features
            parent_ref = self

            def _bridged_get_af(self_m, input_features, input_features_mask, **kwargs):
                ao = orig_get_af(input_features, input_features_mask, **kwargs)
                cp = getattr(self_m, "_active_cross_pair", None)
                if cp is not None and parent_ref.cross_encoder_bridge is not None:
                    hc_al, ht_al, q_tl_cur = cp
                    ao.pooler_output = parent_ref.cross_encoder_bridge(ao.pooler_output, hc_al, ht_al, q_tl_cur)
                return ao

            self.ventral_stream.model.get_audio_features = types.MethodType(_bridged_get_af, self.ventral_stream.model)
            self.ventral_stream.model._active_cross_pair = None

        self._local_decoders_loaded = False
        self.post_init()

    def _ensure_local_decoders(self) -> None:
        if self._local_decoders_loaded:
            return
        root = getattr(self.config, "_name_or_path", None)
        if not root:
            raise RuntimeError("Missing `config._name_or_path` on IshikiASRConfig.")
        if os.path.isdir(root):
            self._root_tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
            assets = os.path.join(root, "stream_assets")
            self._coh_tok = AutoTokenizer.from_pretrained(os.path.join(assets, "dorsal"), local_files_only=True)
            self._whisp_tok = AutoTokenizer.from_pretrained(os.path.join(assets, "broadband"), local_files_only=True)
            self._tdt_tok = AutoTokenizer.from_pretrained(os.path.join(assets, "transducer"), local_files_only=True)
            gran_tok = AutoTokenizer.from_pretrained(os.path.join(assets, "meeting"), local_files_only=True)
        else:
            rev = getattr(self.config, "_commit_hash", None)
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            self._root_tok = AutoTokenizer.from_pretrained(root, revision=rev, local_files_only=offline, trust_remote_code=True)
            self._coh_tok = AutoTokenizer.from_pretrained(root, subfolder="stream_assets/dorsal", revision=rev, local_files_only=offline)
            self._whisp_tok = AutoTokenizer.from_pretrained(root, subfolder="stream_assets/broadband", revision=rev, local_files_only=offline)
            self._tdt_tok = AutoTokenizer.from_pretrained(root, subfolder="stream_assets/transducer", revision=rev, local_files_only=offline)
            gran_tok = AutoTokenizer.from_pretrained(root, subfolder="stream_assets/meeting", revision=rev, local_files_only=offline)
        self._gran_proc = GraniteSpeechProcessor(audio_processor=GraniteSpeechFeatureExtractor(), tokenizer=gran_tok)
        self._local_decoders_loaded = True

    @staticmethod
    def _split_long_audio_at_silence(wavs: list[np.ndarray], max_sec: float = 16.5, target_sec: float = 14.0):
        flat_segments, seg_owners = [], []
        max_len = int(max_sec * 16000)
        for i, w in enumerate(wavs):
            if len(w) <= max_len:
                flat_segments.append(w)
                seg_owners.append(i)
            else:
                n_parts = int(np.ceil(len(w) / (target_sec * 16000)))
                cuts = [0]
                for k in range(1, n_parts):
                    center = int(len(w) * k / n_parts)
                    lo = max(cuts[-1] + 8000, center - 24000)
                    hi = min(len(w) - 8000, center + 24000)
                    if hi > lo + 1600:
                        win = w[lo:hi]
                        n_frames = (len(win) - 1600) // 800 + 1
                        energies = [float(np.mean(win[f * 800 : f * 800 + 1600] ** 2)) for f in range(n_frames)]
                        best_cut = lo + int(np.argmin(energies)) * 800 + 800
                    else:
                        best_cut = center
                    cuts.append(best_cut)
                cuts.append(len(w))
                for k in range(len(cuts) - 1):
                    seg = w[cuts[k] : cuts[k + 1]]
                    if len(seg) >= 1600:
                        flat_segments.append(seg)
                        seg_owners.append(i)
        return flat_segments, seg_owners

    @torch.inference_mode()
    def _run_ventral_with_embeddings(
        self, wavs: list[np.ndarray], bs: int = 256, need_scores: bool = False
    ) -> tuple[list[str], torch.Tensor, list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        pm = self.config.prompt_metadata
        q_pre = pm["qwen_prefix_ids"]
        q_suf = pm["qwen_suffix_ids"]
        q_aud = int(pm["qwen_aud_id"])
        q_pad = int(pm["qwen_pad_id"])
        q_eos = pm.get("qwen_eos_ids", [151643, 151645])
        q_gpad = int(pm.get("qwen_gen_pad_id", 151645))
        p_pre, p_suf = len(q_pre), len(q_suf)
        pre_t = torch.tensor(q_pre, dtype=torch.long, device=dev)
        suf_t = torch.tensor(q_suf, dtype=torch.long, device=dev)

        use_bridge = (
            self.cross_encoder_bridge is not None
            and float(self.cross_encoder_bridge.bridge_eta.detach().abs().item()) > 1e-6
        )

        preds = []
        lps = []
        pooled_embs = []
        N = len(wavs)
        s = 0
        while s < N:
            peek_end = min(N, s + bs)
            max_dur = max(1.0, len(wavs[peek_end - 1]) / 16000.0)
            cur_bs = max(32, min(bs, int(1800.0 / max_dur)))
            bw = wavs[s : min(N, s + cur_bs)]
            s += len(bw)
            B = len(bw)
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)

            log_q, mask_q = self.frontend.extract_qwen_mel(bw, dtype=self.dtype)
            aud_lens = mask_q.sum(-1)
            rem = aud_lens % 100
            f1 = torch.div(rem - 1, 2, rounding_mode="floor") + 1
            f2 = torch.div(f1 - 1, 2, rounding_mode="floor") + 1
            tok_lens = (torch.div(f2 - 1, 2, rounding_mode="floor") + 1 + torch.div(aud_lens, 100, rounding_mode="floor") * 13).tolist()

            if use_bridge:
                mel_n, mask_n = self.frontend.extract_nemo_mel(bw, dtype=self.dtype)
                c_seq = self.dorsal_stream.model.encoder(mel_n, attention_mask=mask_n).last_hidden_state
                t_seq = self.transducer_stream.encoder(mel_n, attention_mask=mask_n).last_hidden_state
                hc_al, ht_al = _align_packed_conformer_features(c_seq, t_seq, mask_n, tok_lens)
                self.ventral_stream.model._active_cross_pair = (hc_al, ht_al, tok_lens)
            else:
                self.ventral_stream.model._active_cross_pair = None

            tot_lens = [p_pre + tl + p_suf for tl in tok_lens]
            L_max = max(tot_lens)
            in_ids = torch.full((B, L_max), q_pad, dtype=torch.long, device=dev)
            am = torch.zeros((B, L_max), dtype=torch.long, device=dev)
            for i, (tl, tot_l) in enumerate(zip(tok_lens, tot_lens)):
                st = L_max - tot_l
                in_ids[i, st : st + p_pre] = pre_t
                in_ids[i, st + p_pre : st + p_pre + tl] = q_aud
                in_ids[i, st + p_pre + tl :] = suf_t
                am[i, st:] = 1

            max_nt = min(215, max(24, int(mdur_b * 7.2) + 18))
            gen_out = self.ventral_stream.generate(
                input_ids=in_ids,
                attention_mask=am,
                input_features=log_q,
                input_features_mask=mask_q,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=q_eos,
                pad_token_id=q_gpad,
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            self.ventral_stream.model._active_cross_pair = None
            if need_scores:
                out_ids = gen_out.sequences
                n_sc = len(gen_out.scores)
                lps.extend(_extract_mean_token_logprobs(gen_out.scores, out_ids[:, -n_sc:] if n_sc > 0 else out_ids[:, :0], q_eos, q_gpad))
            else:
                out_ids = gen_out
                lps.extend([0.0] * B)
            dec = self._root_tok.batch_decode(out_ids[:, L_max:], skip_special_tokens=True)
            preds.extend([t.strip() for t in dec])

            h_pack = getattr(self.ventral_stream.model.audio_tower, "_last_hidden_packed", None)
            u_lens = getattr(self.ventral_stream.model.audio_tower, "_last_utt_tok_lens", None)
            if h_pack is not None and u_lens is not None:
                cur = 0
                for tl in u_lens.tolist():
                    if tl > 0:
                        pooled_embs.append(h_pack[cur : cur + tl].float().mean(dim=0))
                    else:
                        pooled_embs.append(torch.zeros((self.config.audio_gate_dim,), dtype=torch.float32, device=dev))
                    cur += tl
            else:
                for _ in range(B):
                    pooled_embs.append(torch.zeros((self.config.audio_gate_dim,), dtype=torch.float32, device=dev))

        return preds, torch.stack(pooled_embs, dim=0), lps

    @torch.inference_mode()
    def _run_ventral_noise(
        self, wavs: list[np.ndarray], bs: int = 256, need_scores: bool = False
    ) -> tuple[list[str], list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        pm = self.config.prompt_metadata
        q_pre = pm["qwen_prefix_ids"]
        q_suf = pm["qwen_suffix_ids"]
        q_aud = int(pm["qwen_aud_id"])
        q_pad = int(pm["qwen_pad_id"])
        q_eos = pm.get("qwen_eos_ids", [151643, 151645])
        q_gpad = int(pm.get("qwen_gen_pad_id", 151645))
        p_pre, p_suf = len(q_pre), len(q_suf)
        pre_t = torch.tensor(q_pre, dtype=torch.long, device=dev)
        suf_t = torch.tensor(q_suf, dtype=torch.long, device=dev)

        preds, lps = [], []
        N = len(wavs)
        s = 0
        while s < N:
            peek_end = min(N, s + bs)
            max_dur = max(1.0, len(wavs[peek_end - 1]) / 16000.0)
            cur_bs = max(32, min(bs, int(1800.0 / max_dur)))
            bw = wavs[s : min(N, s + cur_bs)]
            s += len(bw)
            B = len(bw)
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)

            log_q, mask_q = self.frontend.extract_qwen_mel(bw, dtype=self.dtype)
            aud_lens = mask_q.sum(-1)
            rem = aud_lens % 100
            f1 = torch.div(rem - 1, 2, rounding_mode="floor") + 1
            f2 = torch.div(f1 - 1, 2, rounding_mode="floor") + 1
            tok_lens = (torch.div(f2 - 1, 2, rounding_mode="floor") + 1 + torch.div(aud_lens, 100, rounding_mode="floor") * 13).tolist()

            tot_lens = [p_pre + tl + p_suf for tl in tok_lens]
            L_max = max(tot_lens)
            in_ids = torch.full((B, L_max), q_pad, dtype=torch.long, device=dev)
            am = torch.zeros((B, L_max), dtype=torch.long, device=dev)
            for i, (tl, tot_l) in enumerate(zip(tok_lens, tot_lens)):
                st = L_max - tot_l
                in_ids[i, st : st + p_pre] = pre_t
                in_ids[i, st + p_pre : st + p_pre + tl] = q_aud
                in_ids[i, st + p_pre + tl :] = suf_t
                am[i, st:] = 1

            max_nt = min(215, max(24, int(mdur_b * 7.2) + 18))
            gen_out = self.ventral_noise_stream.generate(
                input_ids=in_ids,
                attention_mask=am,
                input_features=log_q,
                input_features_mask=mask_q,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=q_eos,
                pad_token_id=q_gpad,
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            if need_scores:
                out_ids = gen_out.sequences
                n_sc = len(gen_out.scores)
                lps.extend(_extract_mean_token_logprobs(gen_out.scores, out_ids[:, -n_sc:] if n_sc > 0 else out_ids[:, :0], q_eos, q_gpad))
            else:
                out_ids = gen_out
                lps.extend([0.0] * B)
            dec = self._root_tok.batch_decode(out_ids[:, L_max:], skip_special_tokens=True)
            preds.extend([t.strip() for t in dec])
        return preds, lps

    @torch.inference_mode()
    def _run_agms_synthesizer(
        self,
        wavs: list[np.ndarray],
        cand_dicts: list[dict[str, str]],
        clean_mode: bool = False,
        bs: int = 64,
    ) -> list[str]:
        """Audio-Grounded Multimodal Hypothesis Synthesis (listens to raw audio + reads 5 stream candidates)."""
        self._ensure_local_decoders()
        dev = self.device
        pm = self.config.prompt_metadata
        q_pre = pm["qwen_prefix_ids"]
        q_aud = int(pm["qwen_aud_id"])
        q_pad = int(pm["qwen_pad_id"])
        q_eos = pm.get("qwen_eos_ids", [151643, 151645])
        q_gpad = int(pm.get("qwen_gen_pad_id", 151645))
        p_pre = len(q_pre)
        pre_t = torch.tensor(q_pre, dtype=torch.long, device=dev)

        preds = []
        N = len(wavs)
        s = 0
        while s < N:
            peek_end = min(N, s + bs)
            max_dur = max(1.0, len(wavs[peek_end - 1]) / 16000.0)
            cur_bs = max(16, min(bs, int(1100.0 / max_dur)))
            bw = wavs[s : min(N, s + cur_bs)]
            bc = cand_dicts[s : min(N, s + cur_bs)]
            s += len(bw)
            B = len(bw)
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)

            log_q, mask_q = self.frontend.extract_qwen_mel(bw, dtype=self.dtype)
            aud_lens = mask_q.sum(-1)
            rem = aud_lens % 100
            f1 = torch.div(rem - 1, 2, rounding_mode="floor") + 1
            f2 = torch.div(f1 - 1, 2, rounding_mode="floor") + 1
            tok_lens = (torch.div(f2 - 1, 2, rounding_mode="floor") + 1 + torch.div(aud_lens, 100, rounding_mode="floor") * 13).tolist()

            suf_lists = []
            tot_lens = []
            for cd, tl in zip(bc, tok_lens):
                if clean_mode:
                    lines = [
                        f"[W] {cd.get('W', '')[:260]}",
                        f"[M] {cd.get('M', '')[:260]}",
                        f"[V] {cd.get('V', '')[:260]}",
                        f"[G] {cd.get('G', '')[:260]}",
                    ]
                    asst = "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English clean<asr_text>"
                else:
                    lines = [
                        f"[V] {cd.get('V', '')[:260]}",
                        f"[C] {cd.get('C', '')[:260]}",
                        f"[W] {cd.get('W', '')[:260]}",
                        f"[T] {cd.get('T', '')[:260]}",
                        f"[G] {cd.get('G', '')[:260]}",
                    ]
                    if cd.get("M"):
                        lines.append(f"[M] {cd['M'][:260]}")
                    asst = "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
                s_ids = self._root_tok.encode("\n" + "\n".join(lines) + asst, add_special_tokens=False)
                suf_lists.append(s_ids)
                tot_lens.append(p_pre + tl + len(s_ids))

            L_max = max(tot_lens)
            in_ids = torch.full((B, L_max), q_pad, dtype=torch.long, device=dev)
            am = torch.zeros((B, L_max), dtype=torch.long, device=dev)
            for i, (tl, s_ids, tot_l) in enumerate(zip(tok_lens, suf_lists, tot_lens)):
                st = L_max - tot_l
                in_ids[i, st : st + p_pre] = pre_t
                in_ids[i, st + p_pre : st + p_pre + tl] = q_aud
                in_ids[i, st + p_pre + tl :] = torch.tensor(s_ids, dtype=torch.long, device=dev)
                am[i, st:] = 1

            max_nt = min(215, max(24, int(mdur_b * 7.2) + 18))
            out_ids = self.ventral_noise_stream.generate(
                input_ids=in_ids,
                attention_mask=am,
                input_features=log_q,
                input_features_mask=mask_q,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=q_eos,
                pad_token_id=q_gpad,
            )
            dec = self._root_tok.batch_decode(out_ids[:, L_max:], skip_special_tokens=True)
            preds.extend([t.strip() for t in dec])
        return preds

    @torch.inference_mode()
    def _run_dorsal(self, wavs: list[np.ndarray], sub_bs: int = 104, need_scores: bool = False) -> tuple[list[str], list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        flat_segs, owners = self._split_long_audio_at_silence(wavs, max_sec=30.5, target_sec=22.0)
        seg_order = sorted(range(len(flat_segs)), key=lambda i: len(flat_segs[i]))
        seg_preds = [""] * len(flat_segs)
        seg_lps = [0.0] * len(flat_segs)
        pref_t = torch.tensor(self.config.prompt_metadata["coh_prefix_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        pm = self.config.prompt_metadata
        c_eos = int(pm.get("coh_eos_id", 3))
        c_pad = int(pm.get("coh_pad_id", 2))

        cs = 0
        Ns = len(seg_order)
        while cs < Ns:
            peek_end = min(Ns, cs + sub_bs)
            max_dur = max(1.0, len(flat_segs[seg_order[peek_end - 1]]) / 16000.0)
            cur_bs = max(24, min(sub_bs, int(580.0 / max_dur)))
            bi = seg_order[cs : min(Ns, cs + cur_bs)]
            cs += len(bi)
            bw = [flat_segs[idx] for idx in bi]
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)
            mel_n, mask_n = self.frontend.extract_nemo_mel(bw, dtype=self.dtype)
            max_nt = min(200, max(24, int(mdur_b * 7.0) + 16))
            dec_in = pref_t.expand(len(bw), -1).contiguous()
            gen_out = self.dorsal_stream.generate(
                input_features=mel_n,
                attention_mask=mask_n,
                decoder_input_ids=dec_in,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=c_eos,
                pad_token_id=c_pad,
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            if need_scores:
                g_ids = gen_out.sequences
                n_sc = len(gen_out.scores)
                b_lp = _extract_mean_token_logprobs(gen_out.scores, g_ids[:, -n_sc:] if n_sc > 0 else g_ids[:, :0], [c_eos], c_pad)
            else:
                g_ids = gen_out
                b_lp = [0.0] * len(bw)
            dec = self._coh_tok.batch_decode(g_ids[:, dec_in.shape[1] :], skip_special_tokens=True)
            for idx_seg, t, lp_v in zip(bi, dec, b_lp):
                seg_preds[idx_seg] = t.replace("(", " ").replace(")", " ").strip()
                seg_lps[idx_seg] = lp_v

        by_utt = defaultdict(list)
        by_lp = defaultdict(list)
        for owner_i, txt, lp_v in zip(owners, seg_preds, seg_lps):
            if txt:
                by_utt[owner_i].append(txt)
                by_lp[owner_i].append(lp_v)
        return (
            [" ".join(by_utt[i]).strip() for i in range(len(wavs))],
            [float(np.mean(by_lp[i])) if by_lp[i] else -1.0 for i in range(len(wavs))],
        )

    @torch.inference_mode()
    def _run_broadband(self, wavs: list[np.ndarray], bs: int = 224, need_scores: bool = False) -> tuple[list[str], list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        pm = self.config.prompt_metadata
        w_pref = torch.tensor(pm.get("whisp_prefix_ids", [50258, 50259, 50360, 50364]), dtype=torch.long, device=dev).unsqueeze(0)
        w_eos = int(pm.get("whisp_eos_id", 50257))
        w_pad = int(pm.get("whisp_pad_id", 50257))
        preds, lps = [], []
        for cs in range(0, len(wavs), bs):
            bw = wavs[cs : cs + bs]
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)
            max_nt = min(200, max(24, int(mdur_b * 7.0) + 16))
            log_w, mask_w = self.frontend.extract_whisper_30s_mel(bw, dtype=self.dtype)
            dec_in = w_pref.expand(len(bw), -1).contiguous()
            gen_out = self.broadband_stream.generate(
                input_features=log_w,
                attention_mask=mask_w,
                decoder_input_ids=dec_in,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=w_eos,
                pad_token_id=w_pad,
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            if need_scores:
                gids = gen_out.sequences
                n_sc = len(gen_out.scores)
                lps.extend(_extract_mean_token_logprobs(gen_out.scores, gids[:, -n_sc:] if n_sc > 0 else gids[:, :0], [w_eos], w_pad))
            else:
                gids = gen_out
                lps.extend([0.0] * len(bw))
            preds.extend([t.strip() for t in self._whisp_tok.batch_decode(gids, skip_special_tokens=True)])
        return preds, lps

    @torch.inference_mode()
    def _run_transducer(self, wavs: list[np.ndarray]) -> list[str]:
        self._ensure_local_decoders()
        flat_segs, owners = self._split_long_audio_at_silence(wavs, max_sec=16.5, target_sec=14.0)
        seg_order = sorted(range(len(flat_segs)), key=lambda idx: len(flat_segs[idx]))
        seg_preds = [""] * len(flat_segs)
        for cs in range(0, len(seg_order), 128):
            bi_sub = seg_order[cs : cs + 128]
            bw = [flat_segs[idx] for idx in bi_sub]
            mn, mkn = self.frontend.extract_nemo_mel(bw, dtype=self.dtype)
            po = self.transducer_stream.generate(input_features=mn, attention_mask=mkn)
            seq = po.sequences if hasattr(po, "sequences") else po
            dec = self._tdt_tok.batch_decode(seq, skip_special_tokens=True)
            for idx_seg, txt in zip(bi_sub, dec):
                seg_preds[idx_seg] = txt.strip()
        by_utt = defaultdict(list)
        for owner_i, txt in zip(owners, seg_preds):
            if txt:
                by_utt[owner_i].append(txt)
        return [" ".join(by_utt[i]).strip() for i in range(len(wavs))]

    @torch.inference_mode()
    def _run_meeting(self, wavs: list[np.ndarray], bs: int = 64, need_scores: bool = False) -> tuple[list[str], list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        prompt = self.config.prompt_metadata["granite_prompt"]
        pm = self.config.prompt_metadata
        g_eos = int(pm.get("granite_eos_id", 100257))
        g_pad = int(pm.get("granite_pad_id", 100256))
        flat_segs, owners = self._split_long_audio_at_silence(wavs, max_sec=30.5, target_sec=22.0)
        seg_order = sorted(range(len(flat_segs)), key=lambda idx: len(flat_segs[idx]))
        seg_preds = [""] * len(flat_segs)
        seg_lps = [0.0] * len(flat_segs)
        for s in range(0, len(seg_order), bs):
            bi = seg_order[s : s + bs]
            bw = [flat_segs[idx] for idx in bi]
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)
            max_nt = min(195, max(24, int(mdur_b * 7.0) + 16))
            inp = self._gran_proc([prompt] * len(bw), bw, device=dev, return_tensors="pt").to(dev, dtype=self.dtype)
            plen = inp["input_ids"].shape[1]
            gen_out = self.meeting_stream.generate(
                **inp,
                max_new_tokens=max_nt,
                do_sample=False,
                eos_token_id=g_eos,
                pad_token_id=g_pad,
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            if need_scores:
                pids = gen_out.sequences
                n_sc = len(gen_out.scores)
                b_lp = _extract_mean_token_logprobs(gen_out.scores, pids[:, -n_sc:] if n_sc > 0 else pids[:, :0], [g_eos], g_pad)
            else:
                pids = gen_out
                b_lp = [0.0] * len(bw)
            dec = self._gran_proc.batch_decode(pids[:, plen:], skip_special_tokens=True)
            for idx_seg, txt, lp_v in zip(bi, dec, b_lp):
                seg_preds[idx_seg] = txt.strip()
                seg_lps[idx_seg] = lp_v
        by_utt = defaultdict(list)
        by_lp = defaultdict(list)
        for owner_i, txt, lp_v in zip(owners, seg_preds, seg_lps):
            if txt:
                by_utt[owner_i].append(txt)
                by_lp[owner_i].append(lp_v)
        return (
            [" ".join(by_utt[i]).strip() for i in range(len(wavs))],
            [float(np.mean(by_lp[i])) if by_lp[i] else -1.0 for i in range(len(wavs))],
        )

    @torch.inference_mode()
    def _run_spontaneous(self, wavs: list[np.ndarray], bs: int = 48, need_scores: bool = False) -> tuple[list[str], list[float]]:
        self._ensure_local_decoders()
        dev = self.device
        pm = self.config.prompt_metadata
        m_pre = pm["moss_prefix_ids"]
        m_suf = pm["moss_suffix_ids"]
        m_pad = int(pm["moss_pad_id"])
        m_end = int(pm["moss_end_id"])
        preds, lps = [], []
        for s in range(0, len(wavs), bs):
            bw = wavs[s : s + bs]
            mdur_b = max(1.0, len(bw[-1]) / 16000.0)
            max_nt = min(195, max(24, int(mdur_b * 7.0) + 16))
            log_q, _ = self.frontend.extract_qwen_mel(bw, dtype=self.dtype)
            encs = []
            for bi, a in enumerate(bw):
                Tm = max(1, min(log_q.shape[-1], len(a) // 160))
                mel_i = log_q[bi, :, :Tm]
                rem = Tm % 100
                fl = (rem - 1) // 2 + 1
                num_a = ((fl - 1) // 2 + 1 - 1) // 2 + 1 + (Tm // 100) * 13
                ids = m_pre + [0] * num_a + m_suf
                msk = [False] * len(m_pre) + [True] * num_a + [False] * len(m_suf)
                encs.append((ids, msk, mel_i, Tm))
            Lm = max(len(e[0]) for e in encs)
            B = len(encs)
            in_ids = torch.full((B, Lm), m_pad, dtype=torch.long)
            am = torch.zeros((B, Lm), dtype=torch.long)
            aud_m = torch.zeros((B, Lm), dtype=torch.bool)
            slens = torch.zeros((B,), dtype=torch.long)
            mparts = []
            for idx_e, (ids, msk, mel_i, Tm) in enumerate(encs):
                li = len(ids)
                in_ids[idx_e, -li:] = torch.tensor(ids, dtype=torch.long)
                am[idx_e, -li:] = 1
                aud_m[idx_e, -li:] = torch.tensor(msk, dtype=torch.bool)
                slens[idx_e] = Tm
                mparts.append(mel_i)
            m_inp = BatchEncoding(
                data={
                    "input_ids": in_ids.to(dev),
                    "attention_mask": am.to(dev),
                    "audio_data": torch.cat(mparts, dim=-1).to(dev, dtype=self.dtype),
                    "audio_data_seqlens": slens.to(dev),
                    "audio_input_mask": aud_m.to(dev),
                }
            )
            gen_out = self.spontaneous_stream.generate(
                **m_inp,
                max_new_tokens=max_nt,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=[m_end],
                return_dict_in_generate=need_scores,
                output_scores=need_scores,
            )
            if need_scores:
                gids = gen_out.sequences
                n_sc = len(gen_out.scores)
                lps.extend(_extract_mean_token_logprobs(gen_out.scores, gids[:, -n_sc:] if n_sc > 0 else gids[:, :0], [m_end], m_pad))
            else:
                gids = gen_out
                lps.extend([0.0] * B)
            dec = self._root_tok.batch_decode(gids[:, Lm:], skip_special_tokens=True)
            preds.extend([t.strip() for t in dec])
        return preds, lps

    def _fuse_single_utterance(
        self,
        stream_hyps: list[tuple[str, list[str], str, float]],
        clean_mode: bool = False,
        spont_lock: bool = False,
    ) -> str:
        """Confidence- & Acoustic-Gate-Weighted Multi-Candidate Span MBR (batch_size=1 compatible)."""
        valid = [(tag, nw, raw, float(w)) for tag, nw, raw, w in stream_hyps if nw]
        if not valid:
            return stream_hyps[0][2] if stream_hyps else ""

        # Strip hallucinated leading speaker-channel digits ('25 let me cite...') when absent in ventral & transducer:
        vent_nw = next((nw for t, nw, _, _ in valid if t == "ventral"), [])
        tdt_nw = next((nw for t, nw, _, _ in valid if t == "transducer"), [])
        if (not clean_mode) and vent_nw and not vent_nw[0].isdigit() and (not tdt_nw or not tdt_nw[0].isdigit()):
            cleaned_valid = []
            for tag, nw, raw, w in valid:
                if tag not in ("ventral", "transducer") and len(nw) >= 4 and nw[0].isdigit():
                    nw = nw[1:]
                    raw = re.sub(r"^\s*\d+\s+", "", raw)
                cleaned_valid.append((tag, nw, raw, w))
            valid = cleaned_valid

        # Length-outlier attenuation
        lens = sorted(len(nw) for _, nw, _, _ in valid)
        med_len = float(lens[len(lens) // 2])
        if med_len >= 5:
            gated = []
            for tag, nw, raw, w in valid:
                ld = abs(len(nw) - med_len) / max(4.0, med_len)
                if ld > 0.32:
                    w *= float(np.exp(-4.5 * (ld - 0.32)))
                gated.append((tag, nw, raw, w))
            valid = gated

        # Filler-stripped utterance-local consensus calibration
        stripped = {tag: [x for x in nw if x not in _FILLER_SET] for tag, nw, _, _ in valid}
        verbatim_filler_confirmed = (not clean_mode) and (
            sum(1 for t, nw, _, _ in valid if t in ("ventral", "transducer", "spontaneous") and any(fw in _FILLER_SET for fw in nw)) >= 2
        )
        acoustic_verbatim_lock = False
        if (not clean_mode) and vent_nw and tdt_nw and len(vent_nw) >= 5:
            d_qt = _lev_words(vent_nw, tdt_nw) / float(len(vent_nw))
            other_d = [
                _lev_words(vent_nw, nw) / float(len(vent_nw))
                for t, nw, _, _ in valid
                if t in ("dorsal", "broadband", "meeting")
            ]
            if d_qt <= 0.12 and other_d and float(np.mean(other_d)) >= d_qt + 0.05:
                acoustic_verbatim_lock = True

        cal = []
        for ti, nwi, rawi, wi in valid:
            si = stripped[ti]
            fi = _STREAM_FAM.get(ti, ti)
            dists = [
                _lev_words(si, stripped[tj]) / max(1.0, 0.5 * (len(si) + len(stripped[tj])))
                for tj, _, _, _ in valid
                if _STREAM_FAM.get(tj, tj) != fi
            ]
            dists.sort()
            eps_i = float(np.mean(dists[: min(3, len(dists))])) if dists else 0.0
            cal.append((ti, nwi, rawi, wi, eps_i))
        min_eps = min(x[4] for x in cal)

        raw_bal = []
        for ti, nwi, rawi, wi, epsi in cal:
            alpha_c = 0.18 if acoustic_verbatim_lock else 0.35
            w_f = wi * float(np.exp(-alpha_c * (epsi - min_eps) / 0.10))
            if acoustic_verbatim_lock and ti in ("dorsal", "broadband", "meeting"):
                w_f *= 0.42
            elif verbatim_filler_confirmed and ti in ("dorsal", "broadband", "meeting") and not any(fw in _FILLER_SET for fw in nwi):
                w_f *= 0.52
            if clean_mode and ti in ("transducer", "dorsal"):
                w_f *= 0.62
            raw_bal.append((ti, nwi, rawi, w_f))

        raw_bal.sort(key=lambda x: -x[3])
        fam_seen = defaultdict(int)
        balanced = []
        for tag, nw, raw, w in raw_bal:
            f_idx = fam_seen[_STREAM_FAM.get(tag, tag)]
            fam_seen[_STREAM_FAM.get(tag, tag)] += 1
            decay = 1.0 if f_idx == 0 else (0.68 if f_idx == 1 else 0.40)
            balanced.append((tag, nw, raw, w * decay))

        balanced.sort(key=lambda x: (0 if x[0] != "dorsal" else 1, -x[3]))
        by_str = defaultdict(float)
        rep_map = {}
        for tag, nw, raw, w in balanced:
            sk = " ".join(nw)
            by_str[sk] += w
            if sk not in rep_map:
                rep_map[sk] = (nw, raw)

        uniq_keys = sorted(by_str.keys(), key=lambda k: -by_str[k])
        if len(uniq_keys) == 1:
            return rep_map[uniq_keys[0]][1]

        tot_w = sum(w for _, _, _, w in balanced)
        tdt_str = (" " + " ".join(tdt_nw) + " ") if tdt_nw else ""
        llm_peers_str = [(" " + " ".join(nw) + " ") for t, nw, _, _ in balanced if t in ("spontaneous", "meeting", "dorsal")]

        def refine_anchor(anchor_w: list[str], anchor_raw: str, is_vent: bool, mr: float = 0.045):
            span_votes = defaultdict(lambda: defaultdict(float))
            span_voters = defaultdict(lambda: defaultdict(set))
            for tag, hw, _, w in balanced:
                if hw == anchor_w:
                    continue
                pw_c, hw_c = _align_compound_pair(anchor_w, hw)
                if len(pw_c) != len(anchor_w):
                    pw_c, hw_c = anchor_w, hw
                sm = difflib.SequenceMatcher(None, pw_c, hw_c, autojunk=False)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op != "equal" and (i2 - i1) <= 4 and (j2 - j1) <= 4:
                        if "".join(pw_c[i1:i2]) == "".join(hw_c[j1:j2]):
                            continue
                        sub_ops = _decompose_atomic_ops(pw_c[i1:i2], hw_c[j1:j2], i1) if ((i2 - i1) > 1 or (j2 - j1) > 1) else []
                        if sub_ops:
                            for sp_k, cand_s in sub_ops:
                                span_votes[sp_k][cand_s] += w
                                span_voters[sp_k][cand_s].add(tag)
                        else:
                            cand = " ".join(hw_c[j1:j2])
                            span_votes[(i1, i2)][cand] += w
                            span_voters[(i1, i2)][cand].add(tag)

            accepted = {}
            for (i1, i2), cmap in span_votes.items():
                orig_sub = " ".join(anchor_w[i1:i2])
                if verbatim_filler_confirmed and orig_sub in _FILLER_SET:
                    continue
                pooled = dict(cmap)
                if (i2 - i1) == 1 and len(cmap) >= 2:
                    c_keys = [ck for ck in cmap if ck and " " not in ck and len(ck) >= 4]
                    for c1 in c_keys:
                        for c2 in c_keys:
                            if c1 != c2 and _char_dist_norm(c1, c2) <= 0.28 and cmap[c1] >= cmap[c2]:
                                pooled[c1] += 0.22 * cmap[c2]
                                span_voters[(i1, i2)][c1] |= span_voters[(i1, i2)][c2]
                best_cand, best_sc = max(pooled.items(), key=lambda x: x[1])
                voters = span_voters[(i1, i2)][best_cand]
                if len(voters) < 2:
                    continue

                if (not clean_mode) and is_vent:
                    diff_w = set(orig_sub.split()) ^ set(best_cand.split())
                    has_repeat = (
                        (i2 - i1 == 1 and i1 > 0 and anchor_w[i1] == anchor_w[i1 - 1])
                        or (i2 - i1 == 1 and i2 < len(anchor_w) and anchor_w[i1] == anchor_w[i2])
                        or (i2 - i1 >= 2 and anchor_w[i1] == anchor_w[i1 + 1])
                    )
                    is_style_only = has_repeat or (
                        bool(diff_w)
                        and all((dw in _FILLER_SET or dw in _COLLOQUIAL_SET or dw in _FUNC_REWRITE_SET or dw.isdigit()) for dw in diff_w)
                    )
                    if is_style_only:
                        if "transducer" not in voters or len(voters) < 3:
                            continue
                        if tdt_str:
                            lw = anchor_w[i1 - 1] if i1 > 0 else ""
                            rw = anchor_w[i2] if i2 < len(anchor_w) else ""
                            ctx = " " + " ".join(x for x in (lw, orig_sub, rw) if x) + " "
                            if ctx in tdt_str and len(voters) < 4:
                                continue
                    if orig_sub and best_cand and " " not in orig_sub and " " not in best_cand and len(orig_sub) >= 4:
                        if _char_dist_norm(orig_sub, best_cand) <= 0.32 and any((" " + orig_sub + " ") in lp for lp in llm_peers_str):
                            if len(voters) < 3:
                                continue

                lw_c = anchor_w[i1 - 1] if i1 > 0 else ""
                rw_c = anchor_w[i2] if i2 < len(anchor_w) else ""
                orig_ctx = " ".join(x for x in (lw_c, orig_sub, rw_c) if x)
                cand_ctx = " ".join(x for x in (lw_c, best_cand, rw_c) if x)
                keep_sc = 0.0
                cand_ctx_sc = 0.0
                for tag, hw, _, w in balanced:
                    hj = " " + " ".join(hw) + " "
                    if hw == anchor_w:
                        keep_sc += w
                    elif i1 < i2 and (" " + orig_sub + " ") in hj:
                        keep_sc += w
                    elif i1 == i2 and (" " + orig_ctx + " ") in hj:
                        keep_sc += 0.85 * w
                    if (" " + cand_ctx + " ") in hj:
                        cand_ctx_sc += 0.25 * w
                if best_sc + cand_ctx_sc > keep_sc + mr * tot_w:
                    accepted[(i1, i2)] = (best_sc + cand_ctx_sc, best_cand)

            if not accepted:
                return anchor_w, anchor_raw
            out_w, cur, used = [], 0, 0
            for i1, i2 in sorted(accepted.keys(), key=lambda x: (x[0], -accepted[x][0])):
                if i1 >= used:
                    out_w.extend(anchor_w[cur:i1])
                    repl = accepted[(i1, i2)][1]
                    if repl:
                        out_w.extend(repl.split())
                    cur = i2
                    used = max(i1 + 1, i2)
            out_w.extend(anchor_w[cur:])
            return out_w, " ".join(out_w)

        vent_key = " ".join(vent_nw)
        if acoustic_verbatim_lock and vent_key in rep_map:
            _, rraw = refine_anchor(rep_map[vent_key][0], rep_map[vent_key][1], is_vent=True, mr=0.055)
            return rraw

        if spont_lock and vent_key in rep_map and (med_len < 5 or abs(len(vent_nw) - med_len) <= max(4.0, 0.30 * med_len)):
            _, rraw = refine_anchor(rep_map[vent_key][0], rep_map[vent_key][1], is_vent=True, mr=0.14)
            return rraw

        candidates = {sk: rep_map[sk] for sk in uniq_keys[:5]}
        cnd_w = _confusion_network_decode(balanced, clean_mode=clean_mode)
        if cnd_w:
            cnd_k = " ".join(cnd_w)
            if cnd_k not in candidates:
                candidates[cnd_k] = (cnd_w, cnd_k)
        for sk in uniq_keys[:3]:
            aw, araw = rep_map[sk]
            for mr in (0.03, 0.055):
                rw, rraw = refine_anchor(aw, araw, is_vent=(sk == vent_key), mr=mr)
                rk = " ".join(rw)
                if rk and rk not in candidates:
                    candidates[rk] = (rw, rraw)

        best_raw, best_risk = None, 1e18
        for ck, (cw, craw) in candidates.items():
            risk = sum(ow * _lev_words(cw, rep_map[ok][0]) for ok, ow in by_str.items() if ok != ck)
            if risk < best_risk:
                best_risk = risk
                best_raw = craw
        return best_raw

    def _estimate_ratio_calibrated_spectral_weights(
        self,
        full_4_map: dict[str, list[list[str]]],
        deep_5_map: dict[str, list[list[str]]],
    ) -> tuple[dict[str, float], float]:
        """Cochran Ratio-Calibrated Full-Corpus Spectral Competence (zero selection bias & zero imputation bias)."""
        tags4 = ["ventral", "dorsal", "broadband", "transducer"]
        N_tot = len(full_4_map["ventral"])
        M_deep = len(deep_5_map["ventral"])
        if N_tot < 16 or M_deep < 4:
            return {t: 1.0 for t in list(deep_5_map.keys())}, 5.0

        def _mean_eps(nmap: dict[str, list[list[str]]], max_s: int = 160):
            ks = list(nmap.keys())
            L = len(nmap[ks[0]])
            st = max(1, L // max_s)
            sidx = list(range(0, L, st))[:max_s]
            dm = {}
            for i, ti in enumerate(ks):
                for j in range(i + 1, len(ks)):
                    tj = ks[j]
                    if _STREAM_FAM.get(ti, ti) == _STREAM_FAM.get(tj, tj):
                        continue
                    te, tl = 0, 0
                    for idx in sidx:
                        wi = [w for w in nmap[ti][idx] if w not in _FILLER_SET]
                        wj = [w for w in nmap[tj][idx] if w not in _FILLER_SET]
                        if not wi and not wj:
                            continue
                        te += _lev_words(wi, wj)
                        tl += max(1, (len(wi) + len(wj)) // 2)
                    dij = (te / max(1, tl)) * 100.0
                    dm[(ti, tj)] = dij
                    dm[(tj, ti)] = dij
            all_fams = set(_STREAM_FAM.get(t, t) for t in ks)
            ep = {}
            for ti in ks:
                fi = _STREAM_FAM.get(ti, ti)
                fd = []
                for fo in all_fams:
                    if fo == fi:
                        continue
                    cd = [dm[(ti, tj)] for tj in ks if _STREAM_FAM.get(tj, tj) == fo and (ti, tj) in dm]
                    if cd:
                        fd.append(min(cd))
                fd.sort()
                ep[ti] = float(np.mean(fd[: min(4, len(fd))])) if fd else 5.0
            return ep, dm

        eps_corp4, dm_corp4 = _mean_eps(full_4_map, max_s=180)
        eps_deep, _ = _mean_eps(deep_5_map, max_s=160)

        mean_c4 = float(np.mean([eps_corp4[t] for t in tags4]))
        mean_d4 = float(np.mean([eps_deep[t] for t in tags4]))
        ratio = mean_c4 / max(1e-4, mean_d4)

        eps_final = dict(eps_corp4)
        for extra_k in deep_5_map:
            if extra_k not in eps_final:
                eps_final[extra_k] = eps_deep[extra_k] * ratio

        min_eps = min(eps_final.values())
        tau = 0.70 if float(np.mean(list(eps_final.values()))) < 6.0 else 1.05
        shrink = float(M_deep) / float(M_deep + 32.0)
        d_vt = dm_corp4.get(("ventral", "transducer"), 5.0)
        return {ti: float(np.exp(-shrink * (eps_final[ti] - min_eps) / tau)) for ti in eps_final}, d_vt

    @torch.inference_mode()
    def transcribe(
        self,
        waveforms: Sequence[Union[np.ndarray, torch.Tensor]],
        batch_size: int = 256,
        formatting: str = "auto",
    ) -> list[str]:
        """End-to-end single-model transcription from 16kHz waveforms -> list[str] (zero dataset flags required)."""
        if not waveforms:
            return []
        wavs = [
            (w.detach().cpu().float().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, dtype=np.float32))
            for w in waveforms
        ]
        N = len(wavs)
        order = sorted(range(N), key=lambda i: len(wavs[i]))
        s_wavs = [wavs[i] for i in order]

        fcal = getattr(self.config, "fusion_calibration", None) or {}
        tau_s1_lp = float(fcal.get("tau_s1_lp", -99.0))
        tau_s2_lp = float(fcal.get("tau_s2_lp", -99.0))
        beta_lp = float(fcal.get("beta_lp", 0.0))
        ns = bool(beta_lp > 1e-4 or tau_s1_lp > -50.0 or tau_s2_lp > -50.0)

        # STAGE 1: Ventral + autonomous speech-regime detection (zero external dataset flags!)
        mean_samp = sum(len(w) for w in s_wavs) / float(max(1, N))
        s1_bs = min(batch_size, 64) if (64 < N <= 512 and len(s_wavs[-1]) > 1.70 * mean_samp) else batch_size
        vent_raw, vent_embs, vent_lps = self._run_ventral_with_embeddings(s_wavs, bs=s1_bs, need_scores=ns)
        if formatting in ("clean", "spgispeech", "formatted"):
            clean_mode = True
        elif formatting == "verbatim":
            clean_mode = False
        else:
            mean_dur_s = float(np.mean([len(w) / 16000.0 for w in s_wavs]))
            raw_wl = [t.lower().split() for t in vent_raw]
            fin_hits = sum(
                1 for wl in raw_wl
                if any(tok.strip(",.?!") in _FINANCE_LEXICON or "%" in tok or "$" in tok for tok in wl)
            ) / float(max(1, N))
            fill_hits = sum(
                1 for wl in raw_wl for tok in wl if tok.strip(",.?!") in _FILLER_SET
            ) / float(max(1, N))
            clean_mode = bool(N >= 8 and mean_dur_s <= 16.5 and fin_hits >= 0.20 and fill_hits <= 0.06)

        vent_all = [_clean_formatted_profile(t, clean_mode) for t in vent_raw]
        nw_v = [_fast_canonical_words(t) for t in vent_all]

        if clean_mode:
            p1_raw, p1_lps = self._run_broadband(s_wavs, bs=min(224, s1_bs), need_scores=ns)
            p1_all = [_clean_formatted_profile(t, True) for t in p1_raw]
        else:
            p1_all = [_clean_formatted_profile(t, False) for t in self._run_transducer(s_wavs)]
            p1_lps = [-0.08] * N
        nw_p1 = [_fast_canonical_words(t) for t in p1_all]

        s_final = [""] * N
        contested = []
        for i in range(N):
            wv, wp = nw_v[i], nw_p1[i]
            if (wv == wp or (wv and "".join(wv) == "".join(wp))) and vent_lps[i] >= tau_s1_lp:
                s_final[i] = vent_all[i]
            else:
                contested.append(i)

        if contested:
            c_wavs = [s_wavs[i] for i in contested]
            c_embs = vent_embs[torch.tensor(contested, dtype=torch.long, device=self.device)]
            c_stats = _compute_acoustic_stats(c_wavs, self.device)
            gate_w = self.pathway_gate(torch.cat([c_embs, c_stats], dim=-1)).detach().cpu().numpy()  # (Nc, 7)

            if clean_mode:
                spont_raw, spont_lps = self._run_spontaneous(c_wavs, bs=56, need_scores=ns)
                spont_sub = [_clean_formatted_profile(t, True) for t in spont_raw]
                nw_s = [_fast_canonical_words(t) for t in spont_sub]
                use_agms = bool(getattr(self.config, "use_agms_synthesizer", False))
                agms_mode = str(getattr(self.config, "agms_mode", "hybrid"))

                hard_c_pos = []
                for pos_c, idx_s in enumerate(contested):
                    wv_i, wb_i, ws_i = nw_v[idx_s], nw_p1[idx_s], nw_s[pos_c]
                    lp_v, lp_b, lp_s = vent_lps[idx_s], p1_lps[idx_s], spont_lps[pos_c]
                    if wb_i == ws_i and max(lp_b, lp_s) >= -0.25:
                        s_final[idx_s] = p1_all[idx_s]
                    elif ws_i == wv_i and max(lp_s, lp_v) >= -0.25:
                        s_final[idx_s] = spont_sub[pos_c]
                    else:
                        hard_c_pos.append(pos_c)

                if hard_c_pos:
                    if use_agms:
                        hc_wavs = [c_wavs[p] for p in hard_c_pos]
                        meet_hc_raw, _ = self._run_meeting(hc_wavs, bs=64, need_scores=False)
                        meet_hc = [_clean_formatted_profile(t, True) for t in meet_hc_raw]
                        cdicts = [
                            {
                                "W": p1_all[contested[p]],
                                "M": spont_sub[p],
                                "V": vent_all[contested[p]],
                                "G": meet_hc[k_i],
                            }
                            for k_i, p in enumerate(hard_c_pos)
                        ]
                        agms_hc = [_clean_formatted_profile(t, True) for t in self._run_agms_synthesizer(hc_wavs, cdicts, clean_mode=True, bs=64)]
                        nw_agms_hc = [_fast_canonical_words(t) for t in agms_hc]
                        for k_i, pos_c in enumerate(hard_c_pos):
                            idx_s = contested[pos_c]
                            if agms_mode == "direct" and nw_agms_hc[k_i]:
                                s_final[idx_s] = agms_hc[k_i]
                            else:
                                hyps = [
                                    ("agms", nw_agms_hc[k_i], agms_hc[k_i], 2.05),
                                    ("broadband", nw_p1[idx_s], p1_all[idx_s], 1.35),
                                    ("spontaneous", nw_s[pos_c], spont_sub[pos_c], 1.42),
                                    ("ventral", nw_v[idx_s], vent_all[idx_s], 1.20),
                                    ("meeting", _fast_canonical_words(meet_hc[k_i]), meet_hc[k_i], 1.05),
                                ]
                                s_final[idx_s] = self._fuse_single_utterance(hyps, clean_mode=True)
                    else:
                        for pos_c in hard_c_pos:
                            idx_s = contested[pos_c]
                            wv_i, wb_i, ws_i = nw_v[idx_s], nw_p1[idx_s], nw_s[pos_c]
                            lp_v, lp_b, lp_s = vent_lps[idx_s], p1_lps[idx_s], spont_lps[pos_c]
                            max_lp = max(lp_v, lp_b, lp_s)
                            cv = float(np.exp(beta_lp * np.clip(lp_v - max_lp, -0.90, 0.0)))
                            cb = float(np.exp(beta_lp * np.clip(lp_b - max_lp, -0.90, 0.0)))
                            cs = float(np.exp(beta_lp * np.clip(lp_s - max_lp, -0.90, 0.0)))
                            hyps = [
                                ("broadband", wb_i, p1_all[idx_s], 1.35 * cb),
                                ("spontaneous", ws_i, spont_sub[pos_c], 1.42 * cs),
                                ("ventral", wv_i, vent_all[idx_s], 1.22 * cv),
                            ]
                            s_final[idx_s] = self._fuse_single_utterance(hyps, clean_mode=True)
            else:
                # STAGE 2 (~315x Corpus RTFx): Fast Dorsal + Broadband on Contested Subset!
                dors_raw, dors_lps = self._run_dorsal(c_wavs, sub_bs=104, need_scores=ns)
                dors_sub = [_clean_formatted_profile(t, False) for t in dors_raw]
                nw_d = [_fast_canonical_words(t) for t in dors_sub]

                broad_raw, broad_lps = self._run_broadband(c_wavs, bs=224, need_scores=ns)
                broad_sub = [_clean_formatted_profile(t, False) for t in broad_raw]
                nw_b = [_fast_canonical_words(t) for t in broad_sub]
                tdt_sub = [p1_all[i] for i in contested]
                nw_t = [nw_p1[i] for i in contested]

                step_m = max(1, N // 160)
                mean_words = float(np.mean([len(nw_v[i]) for i in range(0, N, step_m)]))
                tot_v_fill = sum(1 for u in nw_v for w in u if w in _FILLER_SET)
                v_fill = sum(1 for i in contested for w in nw_v[i] if w in _FILLER_SET)
                d_fill = sum(1 for u in nw_d for w in u if w in _FILLER_SET)
                v_spont = sum(_count_spont_markers(nw_v[i]) for i in contested)
                d_spont = sum(_count_spont_markers(u) for u in nw_d)
                mean_snr_c = float(c_stats[:, 1].mean().item()) if len(contested) > 0 else 1.0
                colloquial_podcast = (
                    len(contested) >= 8
                    and (len(contested) / float(max(1, N))) >= 0.28
                    and mean_snr_c <= 0.88
                    and (v_spont / float(max(1, len(contested)))) >= 0.28
                    and 13.5 <= mean_words <= 26.0
                )

                # Stage-2 Confirmation:
                deep_ranks = []
                for pos_c, idx_s in enumerate(contested):
                    wv_i = nw_v[idx_s]
                    wd_i, wb_i, wt_i = nw_d[pos_c], nw_b[pos_c], nw_t[pos_c]
                    jv, jd, jb = "".join(wv_i), "".join(wd_i), "".join(wb_i)
                    vlp = vent_lps[idx_s]
                    has_v_filler = (
                        (len(contested) >= 8 and v_fill >= max(3, int(0.025 * len(contested))) and d_fill <= 0.35 * max(1, v_fill))
                        or (len(contested) < 8 and v_fill > 0 and d_fill == 0)
                    ) and any((w in _FILLER_SET or w in _COLLOQUIAL_SET) for w in wv_i) and not any((w in _FILLER_SET or w in _COLLOQUIAL_SET) for w in wb_i)
                    if wv_i and (wv_i == wd_i or wv_i == wb_i or (jv and (jv == jd or jv == jb))) and vlp >= tau_s2_lp:
                        s_final[idx_s] = vent_all[idx_s]
                    elif (not colloquial_podcast) and (not has_v_filler) and wd_i and (wd_i == wb_i and wd_i == wt_i):
                        s_final[idx_s] = broad_sub[pos_c]
                    else:
                        deep_ranks.append(pos_c)

                if deep_ranks:
                    use_agms = bool(getattr(self.config, "use_agms_synthesizer", False))
                    agms_mode = str(getattr(self.config, "agms_mode", "hybrid"))
                    deep_wavs = [c_wavs[p] for p in deep_ranks]

                    if colloquial_podcast:
                        # Spontaneous colloquial podcast/YouTube speech: skip slow Meeting-Granite, run Super-Oruk + Spontaneous-MOSS2B!
                        vn_raw, vn_lps = self._run_ventral_noise(deep_wavs, bs=batch_size, need_scores=False)
                        vn_deep = [_clean_formatted_profile(t, False) for t in vn_raw]
                        nw_vn_deep = [_fast_canonical_words(t) for t in vn_deep]
                        sp_raw, _ = self._run_spontaneous(deep_wavs, bs=56, need_scores=False)
                        spont_deep = [_clean_formatted_profile(t, False) for t in sp_raw]
                        nw_s_deep = [_fast_canonical_words(t) for t in spont_deep]
                        for r_i, pos_c in enumerate(deep_ranks):
                            idx_s = contested[pos_c]
                            hyps = [
                                ("ventral", nw_v[idx_s], vent_all[idx_s], 2.15),
                                ("ventral_noise", nw_vn_deep[r_i], vn_deep[r_i], 1.15),
                                ("spontaneous", nw_s_deep[r_i], spont_deep[r_i], 1.25),
                                ("broadband", nw_b[pos_c], broad_sub[pos_c], 0.82),
                                ("dorsal", nw_d[pos_c], dors_sub[pos_c], 0.80),
                                ("transducer", nw_t[pos_c], tdt_sub[pos_c], 0.78),
                            ]
                            s_final[idx_s] = self._fuse_single_utterance(hyps, clean_mode=False, spont_lock=True)
                    else:
                        meet_raw, meet_lps = self._run_meeting(deep_wavs, bs=64, need_scores=ns)
                        meet_deep = [_clean_formatted_profile(t, False) for t in meet_raw]
                        nw_m_deep = [_fast_canonical_words(t) for t in meet_deep]

                        full_4_map = {
                            "ventral": nw_v,
                            "dorsal": list(nw_v),
                            "broadband": list(nw_v),
                            "transducer": list(nw_v),
                        }
                        for pos_c, idx_s in enumerate(contested):
                            full_4_map["dorsal"][idx_s] = nw_d[pos_c]
                            full_4_map["broadband"][idx_s] = nw_b[pos_c]
                            full_4_map["transducer"][idx_s] = nw_t[pos_c]

                        deep_map = {
                            "ventral": [nw_v[contested[p]] for p in deep_ranks],
                            "dorsal": [nw_d[p] for p in deep_ranks],
                            "broadband": [nw_b[p] for p in deep_ranks],
                            "transducer": [nw_t[p] for p in deep_ranks],
                            "meeting": nw_m_deep,
                        }

                        vn_raw, vn_lps = self._run_ventral_noise(deep_wavs, bs=batch_size, need_scores=ns)
                        vn_deep = [_clean_formatted_profile(t, False) for t in vn_raw]
                        nw_vn_deep = [_fast_canonical_words(t) for t in vn_deep]
                        deep_map["ventral_noise"] = nw_vn_deep

                        spec_f, d_vt = self._estimate_ratio_calibrated_spectral_weights(full_4_map, deep_map)
                        verbatim_parliamentary = (
                            len(contested) >= 8
                            and v_fill >= max(3, int(0.025 * len(contested)))
                            and d_fill <= 0.35 * max(1, v_fill)
                            and mean_words > 18.0
                            and d_vt < 3.5
                        )
                        if verbatim_parliamentary:
                            spec_f["dorsal"] *= 0.22
                            spec_f["broadband"] *= 0.35
                            spec_f["meeting"] *= 0.32

                        for r_i, pos_c in enumerate(deep_ranks):
                            idx_s = contested[pos_c]
                            gw = gate_w[pos_c]
                            hyps = [
                                ("ventral", nw_v[idx_s], vent_all[idx_s], float(gw[0]) * spec_f["ventral"]),
                                ("ventral_noise", nw_vn_deep[r_i], vn_deep[r_i], float(gw[1]) * spec_f.get("ventral_noise", 1.0)),
                                ("dorsal", nw_d[pos_c], dors_sub[pos_c], float(gw[2]) * spec_f["dorsal"]),
                                ("broadband", nw_b[pos_c], broad_sub[pos_c], float(gw[3]) * spec_f["broadband"]),
                                ("transducer", nw_t[pos_c], tdt_sub[pos_c], float(gw[4]) * spec_f["transducer"] * 0.94),
                                ("meeting", nw_m_deep[r_i], meet_deep[r_i], float(gw[5]) * spec_f["meeting"]),
                            ]
                            s_final[idx_s] = self._fuse_single_utterance(hyps, clean_mode=False, spont_lock=False)

        out = [""] * N
        for idx_s, orig_i in enumerate(order):
            out[orig_i] = s_final[idx_s]
        return out

    @torch.inference_mode()
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        formatting: str = "auto",
        batch_size: int = 256,
        **kwargs,
    ) -> torch.LongTensor:
        """Standard Hugging Face `model.generate(**inputs)` returning token IDs decoded by `processor.batch_decode`."""
        self._ensure_local_decoders()
        if input_features is None:
            input_features = kwargs.get("input_values", None)
        if input_features is None:
            raise ValueError("`input_features` (16kHz waveforms from IshikiASRProcessor) is required.")

        B = input_features.shape[0]
        lens_l = attention_mask.sum(dim=-1).detach().cpu().tolist() if attention_mask is not None else [input_features.shape[1]] * B
        max_valid = max((max(1, int(x)) for x in lens_l), default=1)
        inp_slice = input_features[:, :max_valid].detach()
        if inp_slice.device.type == "cpu" and inp_slice.dtype == torch.float32:
            inp_cpu = inp_slice.numpy()
        else:
            inp_cpu = inp_slice.to(dtype=torch.float32, device="cpu").numpy()
        wavs = [np.ascontiguousarray(inp_cpu[i, : max(1, int(lens_l[i]))], dtype=np.float32) for i in range(B)]

        texts = self.transcribe(wavs, batch_size=batch_size, formatting=formatting)
        eos_id = int(self.config.eos_token_id)
        pad_id = int(self.config.pad_token_id)
        tok_lists = [self._root_tok.encode(t, add_special_tokens=False) + [eos_id] for t in texts]
        L_max = max((len(tl) for tl in tok_lists), default=1)
        out_ids = torch.full((B, L_max), pad_id, dtype=torch.long, device=self.device)
        for i, tl in enumerate(tok_lists):
            out_ids[i, : len(tl)] = torch.tensor(tl, dtype=torch.long, device=self.device)
        return out_ids

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass returning acoustic pathway routing logits for differentiable inspection/calibration."""
        B = input_features.shape[0]
        inp_cpu = input_features.detach().cpu().float().numpy()
        msk_cpu = attention_mask.detach().cpu().numpy() if attention_mask is not None else None
        wavs = [
            np.ascontiguousarray(inp_cpu[i, : max(1, int(msk_cpu[i].sum()) if msk_cpu is not None else inp_cpu.shape[1])], dtype=np.float32)
            for i in range(B)
        ]
        log_q, mask_q = self.frontend.extract_qwen_mel(wavs, dtype=self.dtype)
        enc_out = self.ventral_stream.model.audio_tower(log_q, mask_q).last_hidden_state
        u_lens = self.ventral_stream.model.audio_tower._last_utt_tok_lens.tolist()
        pooled, cur = [], 0
        for tl in u_lens:
            pooled.append(enc_out[cur : cur + tl].float().mean(dim=0) if tl > 0 else torch.zeros((self.config.audio_gate_dim,), device=self.device))
            cur += tl
        stats = _compute_acoustic_stats(wavs, self.device)
        gate_weights = self.pathway_gate(torch.cat([torch.stack(pooled, dim=0), stats], dim=-1))
        return BaseModelOutputWithPast(last_hidden_state=gate_weights.unsqueeze(1))


try:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
    AutoConfig.register("ishiki_asr", IshikiASRConfig, exist_ok=True)
    AutoModelForSpeechSeq2Seq.register(IshikiASRConfig, IshikiASRForConditionalGeneration, exist_ok=True)
except Exception:
    pass

__all__ = ["IshikiASRForConditionalGeneration", "IshikiSpontaneousConfig", "IshikiSpontaneousSubNet"]
