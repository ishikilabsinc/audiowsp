"""Ishiki-ASR Standalone Processor (`processing_ishiki_asr.py`).

Provides standard off-the-shelf Hugging Face `AutoProcessor` interface:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt")
    ids = model.generate(**inputs)
    texts = processor.batch_decode(ids, skip_special_tokens=True)
"""
from __future__ import annotations

import json
import os
from typing import Sequence, Union

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin


class IshikiASRFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )

    def __call__(
        self,
        raw_speech: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        sampling_rate: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        if isinstance(raw_speech, (np.ndarray, torch.Tensor)):
            if raw_speech.ndim == 1:
                speech_list = [raw_speech]
            else:
                speech_list = [raw_speech[i] for i in range(raw_speech.shape[0])]
        else:
            speech_list = list(raw_speech)

        arrays = []
        for s in speech_list:
            if isinstance(s, torch.Tensor):
                a = s.detach().cpu().float().numpy()
            else:
                a = np.asarray(s, dtype=np.float32)
            if a.ndim > 1:
                a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=-1)
            arrays.append(np.ascontiguousarray(a, dtype=np.float32))

        B = len(arrays)
        max_len = max((len(a) for a in arrays), default=1)
        max_len = max(1, max_len)

        padded = np.zeros((B, max_len), dtype=np.float32)
        mask = np.zeros((B, max_len), dtype=np.int64)
        for i, a in enumerate(arrays):
            L = len(a)
            if L > 0:
                padded[i, :L] = a
                mask[i, :L] = 1

        if return_tensors == "pt":
            return BatchFeature(
                data={
                    "input_features": torch.from_numpy(padded),
                    "attention_mask": torch.from_numpy(mask),
                }
            )
        return BatchFeature(data={"input_features": padded, "attention_mask": mask})


class IshikiASRProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor: IshikiASRFeatureExtractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        fe = IshikiASRFeatureExtractor(sampling_rate=16000)
        tok_kwargs = {
            k: kwargs[k]
            for k in ("cache_dir", "force_download", "local_files_only", "proxies", "revision", "subfolder", "token")
            if k in kwargs
        }
        tok_kwargs["trust_remote_code"] = kwargs.get("trust_remote_code", True)
        tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tok_kwargs)
        return cls(feature_extractor=fe, tokenizer=tok)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        cfg = {
            "processor_class": "IshikiASRProcessor",
            "feature_extractor_type": "IshikiASRFeatureExtractor",
            "sampling_rate": 16000,
            "feature_size": 1,
            "padding_value": 0.0,
            "auto_map": {
                "AutoProcessor": "processing_ishiki_asr.IshikiASRProcessor",
            },
        }
        with open(os.path.join(save_directory, "preprocessor_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(save_directory, "processor_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def __call__(
        self,
        audio=None,
        raw_speech=None,
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        speech = audio if audio is not None else raw_speech
        if speech is None:
            raise ValueError("Provide `audio` or `raw_speech` waveform array(s) at 16kHz.")
        return self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors=return_tensors)

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **kwargs) -> list[str]:
        return [t.strip() for t in self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens, **kwargs)]

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        if isinstance(token_ids, (torch.Tensor, np.ndarray)) and token_ids.ndim >= 2:
            return self.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        if isinstance(token_ids, (list, tuple)) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple, torch.Tensor, np.ndarray)):
            return self.batch_decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        res = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        if isinstance(res, list):
            return [t.strip() for t in res]
        return res.strip()


try:
    AutoProcessor.register("ishiki_asr", IshikiASRProcessor)
except Exception:
    pass

__all__ = ["IshikiASRFeatureExtractor", "IshikiASRProcessor"]
