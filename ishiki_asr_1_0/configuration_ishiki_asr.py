"""Ishiki-ASR Standalone Configuration v2.0 (`configuration_ishiki_asr.py`).

Self-contained `PretrainedConfig` for the standalone Ishiki Labs ASR model (`ishiki_asr`).
Zero external repository references or network calls.
"""
from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class IshikiASRConfig(PretrainedConfig):
    model_type = "ishiki_asr"
    is_composition = True

    def __init__(
        self,
        ventral_config: dict | None = None,
        ventral_noise_config: dict | None = None,
        dorsal_config: dict | None = None,
        broadband_config: dict | None = None,
        transducer_config: dict | None = None,
        meeting_config: dict | None = None,
        spontaneous_config: dict | None = None,
        stream_generation_configs: dict | None = None,
        prompt_metadata: dict | None = None,
        use_cross_encoder_bridge: bool = True,
        cross_bridge_hidden: int = 2560,
        use_agms_synthesizer: bool = False,
        agms_mode: str = "hybrid",
        fusion_calibration: dict | None = None,
        audio_gate_dim: int = 1024,
        audio_gate_hidden: int = 256,
        sampling_rate: int = 16000,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 151643,
        eos_token_id: int = 151645,
        bos_token_id: int = 151644,
        **kwargs,
    ):
        kwargs["tie_word_embeddings"] = tie_word_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        self.ventral_config = ventral_config or {}
        self.ventral_noise_config = ventral_noise_config or dict(self.ventral_config)
        self.dorsal_config = dorsal_config or {}
        self.broadband_config = broadband_config or {}
        self.transducer_config = transducer_config or {}
        self.meeting_config = meeting_config or {}
        self.spontaneous_config = spontaneous_config or {}
        self.stream_generation_configs = stream_generation_configs or {}
        self.prompt_metadata = prompt_metadata or {}
        self.use_cross_encoder_bridge = use_cross_encoder_bridge
        self.cross_bridge_hidden = cross_bridge_hidden
        self.use_agms_synthesizer = use_agms_synthesizer
        self.agms_mode = agms_mode
        self.fusion_calibration = fusion_calibration or {
            "tau_s1_lp": -0.18,
            "tau_s2_lp": -0.22,
            "beta_lp": 1.30,
            "margin_low": 0.03,
            "margin_high": 0.055,
        }
        self.audio_gate_dim = audio_gate_dim
        self.audio_gate_hidden = audio_gate_hidden
        self.sampling_rate = sampling_rate


try:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
    from .modeling_ishiki_asr import IshikiASRForConditionalGeneration
    AutoConfig.register("ishiki_asr", IshikiASRConfig, exist_ok=True)
    AutoModelForSpeechSeq2Seq.register(IshikiASRConfig, IshikiASRForConditionalGeneration, exist_ok=True)
except Exception:
    pass

__all__ = ["IshikiASRConfig"]
