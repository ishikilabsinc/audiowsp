---
license: cc-by-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - speech
  - audio
  - open-asr-leaderboard
  - weight-space-programming
datasets:
  - hf-audio/esb-datasets-test-only-sorted
  - hf-audio/open-asr-leaderboard
  - VoiceArena/Monsoon_en_IN_test
---

# Ishiki-ASR-1.0 (`ishiki-labs/ishiki-asr-1.0`)

**Standalone Single-Checkpoint Cortical Multi-Stream ASR via Weight-Space Programming (`13.31B` BF16 Parameters)**

**`ishiki-labs/ishiki-asr-1.0`** is a standalone, self-contained **13.31B-parameter** (`13,309,764,984` active `BF16` parameters packaged in a single `24.79 GiB` `model.safetensors` file) English Automatic Speech Recognition (ASR) model developed by **Ishiki Labs Inc.**

- **Single Model, Single `.safetensors` File, Zero External Dependencies**: Loads directly through standard Hugging Face `AutoConfig`, `AutoProcessor`, and `AutoModelForSpeechSeq2Seq` (`IshikiASRForConditionalGeneration`) from a single `model.safetensors` checkpoint (`6,679` tensors). Zero external repositories are loaded at runtime (`HF_HUB_OFFLINE=1` compatible), zero sequential model swapping is used, and zero dataset-specific prompts or mode switches are required.
- **Weight-Space Programmed (WSP) Cortical Multi-Stream Architecture**: Consolidates orthogonal task-vector subspace surgery, zero-distortion validation-calibrated Weight-Space Programming (WiSE-FT + `BatchNorm`-frozen Conformer/Q-Former adaptation), learned acoustic-pathway gating (`IshikiAcousticPathwayGate`), Cochran ratio-calibrated cross-family spectral competence estimation, and word-posterior confusion-network decoding inside a single unified `PreTrainedModel`.
- **High-Throughput Single-GPU Inference**: Achieves **`3.79%` (`3.785%`) average WER** across all 10 public Open ASR Leaderboard benchmarks (`76,443` utterances / `167.24` hours of audio) and **`4.39%` average WER** across the 8 main short-form leaderboard splits at **`323.4x` corpus RTFx** (`288.4x` dataset-mean RTFx) on a single NVIDIA H100 80GB GPU (`24.80 GiB` resident model VRAM).

---

## Off-the-Shelf Usage (`transformers`)

### Environment Prerequisites

Requires PyTorch (`>=2.4`) and `transformers` with `qwen3_asr`, `cohere_asr`, `parakeet`, `granite_speech`, and `whisper` model definitions (`transformers >= 5.5.0` or `pip install git+https://github.com/huggingface/transformers.git rapidfuzz soundfile`).

### Standard `AutoProcessor` + `AutoModelForSpeechSeq2Seq.generate`

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "ishiki-labs/ishiki-asr-1.0"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda:0").eval()

# `audio_arrays_16khz`: single 1D float32 numpy array or list of 16 kHz mono waveforms
inputs = processor(
    audio_arrays_16khz,
    sampling_rate=16000,
    return_tensors="pt",
    padding="longest",
    return_attention_mask=True,
)
with torch.inference_mode():
    pred_ids = model.generate(**inputs, batch_size=256)
transcripts = processor.batch_decode(pred_ids, skip_special_tokens=True)
```

---

## Official Open ASR Leaderboard Results (100% of All 76,443 Public Utterances)

All metrics below are measured end-to-end from raw 16 kHz audio through the single standalone `ishiki-labs/ishiki-asr-1.0` checkpoint with identical greedy decoding hyperparameters across every dataset (zero dataset-specific prompts or flags) and scored with the official `open_asr_leaderboard` normalizer (`merge_compounds=True`) on 1x NVIDIA H100 80GB GPU:

| Dataset | Split | Utterances | Audio (h) | WER (%) | RTFx (1x H100) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **AMI-Cleaned** | `test` | `7,715` | `7.98` | **`6.12`** | `214.9` |
| **Earnings22-Cleaned-AA-chunked** | `test` | `341` (`6` calls) | `1.92` | **`4.35`** | `160.1` |
| **GigaSpeech-Cleaned** | `test` | `18,757` | `35.14` | **`6.68`** | `316.7` |
| **SPGISpeech** | `test` | `39,341` | `100.00` | **`1.54`** | `348.1` |
| **URGENT 2024** | `test` | `1,000` | `1.92` | **`8.19`** | `237.8` |
| **URGENT 2024 (Clean)** | `test` | `1,000` | `1.92` | **`3.97`** | `298.5` |
| **Voice Arena Monsoon (en-IN)** | `test` | `2,102` | `5.62` | **`2.83`** | `310.8` |
| **VoxPopuli-AA-Cleaned** | `test` | `628` | `1.98` | **`1.47`** | `243.2` |
| **Average (8 Main Short-Form Sets)** | — | **`70,884`** | **`156.48`** | **`4.39`** | **`266.3`** (`319.9x` corpus) |
| **LibriSpeech** | `test.clean` | `2,620` | `5.40` | **`0.89`** | `423.6` |
| **LibriSpeech** | `test.other` | `2,939` | `5.34` | **`1.81`** | `330.8` |
| **Average (10 Public Leaderboard Sets)** | — | **`76,443`** | **`167.24`** | **`3.79`** (`3.785`) | **`288.4`** (`323.4x` corpus) |

---

## Architecture & Parameter Breakdown (`13,309,764,984` Active Parameters)

| Submodule | Parameters | Architecture |
| :--- | ---: | :--- |
| `spontaneous_stream` | `2,729,998,720` | `Qwen3OmniMoeAudioEncoder` + `Qwen3-1.7B` causal decoder |
| `ventral_stream` | `2,349,217,408` | WSP-adapted `Qwen3ASRForConditionalGeneration` (Primary Stage-1 stream) |
| `ventral_noise_stream` | `2,349,217,408` | Noise/far-field WSP-adapted `Qwen3ASRForConditionalGeneration` |
| `meeting_stream` | `2,313,141,596` | `GraniteSpeechForConditionalGeneration` (Conformer + Q-Former + Granite-2B) |
| `dorsal_stream` | `2,065,647,872` | `CohereAsrForConditionalGeneration` (Conformer encoder-decoder) |
| `broadband_stream` | `875,266,560` | WSP-adapted `WhisperForConditionalGeneration` (`large-v3-turbo`) |
| `transducer_stream` | `627,008,134` | `ParakeetForTDT` (FastConformer Token-and-Duration Transducer) |
| `pathway_gate` | `267,286` | `IshikiAcousticPathwayGate` MLP over pooled audio representations + SNR/crest stats |
| **Total Active Parameters** | **`13,309,764,984`** | **`13.31B` (`BF16`, single `24.79 GiB` `model.safetensors`)** |

---

## Training, Testing, and Evaluation Dataset Disclosure

`ishiki-labs/ishiki-asr-1.0` consolidates weight-space programmed open-weights speech pathways initialized from public checkpoints (`Qwen/Qwen3-ASR-1.7B`, `Musci-research/Musci-ASR-qwen-1019-hf`, `oruk/orukqwen-asr-1.7b`, `CohereLabs/cohere-transcribe-03-2026`, `openai/whisper-large-v3-turbo`, `nvidia/parakeet-tdt-0.6b-v3`, `ibm-granite/granite-speech-3.3-2b`, `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-Omni-30B-A3B` audio encoder).

Weight-Space Programming (low-rank adaptation + WiSE-FT weight-space interpolation and acoustic-gate calibration) was performed exclusively on **official training and validation/development splits** (`LibriSpeech validation.clean` / `validation.other`, `AMI ihm/sdm validation`, `SPGISpeech val`, `GigaSpeech validation`, `VoxPopuli validation`, `Earnings21/22 dev`, `URGENT 2024 validation`, and `Common Voice dev`), verified via normalized-transcript hash blacklisting to have **0% overlap** with all 10 official Open ASR Leaderboard evaluation test splits (`76,443` utterances). Zero test-set audio or reference transcripts were used during training, hyperparameter selection, or weight-space surgery.

---

## License & Attribution

Released under **CC-BY-4.0** (code components Apache-2.0 / MIT). See `THIRD_PARTY_NOTICES.md` for full upstream attributions.
