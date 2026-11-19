# AudioWSP: Weight-Space Programming & One-Forward-Pass Optimization for Audio Models

[**Interactive Research Blogpost**](https://aksy1999.github.io/blog/weight-space-programming-for-audio-models/) | [**Audio Deepfake Model (`ishiki-deepfake-0.5`, 86.8M / 318M)**](https://huggingface.co/ishiki-labs/ishiki-deepfake-0.5) | [**Speech Recognition Model (`ishiki-asr-0.5`)**](https://huggingface.co/ishiki-labs/ishiki-asr-0.5) | [**Calibration & Evaluation Dataset (`audiowsp-asr-calibration-data`)**](https://huggingface.co/datasets/ishiki-labs/audiowsp-asr-calibration-data)

`audiowsp` (`harness/audiowsp.py`) is the unified open-source Python repository accompanying **"Weight-Space Programming for Audio Models: LLMs Can Directly Edit Audio Model Weights and Compress Them"** by **Amit Kumar Singh Yadav** and **Robert Xu** (*Ishiki Labs Research*, Nov 2026).

Every reproducible experiment, figure generator, weight-surgery primitive, closed-form optimizer, and evaluation benchmark from the blogpost lives in this single repository.

---

## Five Core Building Blocks (`harness/audiowsp.py`)

All five building blocks execute inside `torch.inference_mode()` (`1 Forward Pass + 0 Backward Passes`) and are guarded by `harness/firewall.py` (`EvalFirewallViolation`) to prevent any evaluation-split access during calibration:

1. **Block 1 — Benchmarking Weight Sensitivity to Audio Augmentations (`AcousticNuisanceProfiler`)**: Synthesizes frame-aligned paired clean vs. corrupted probes (`.WAV` vs. `.MP3`, `Encodec`, `8 kHz` PSTN $\mu$-law, RIR reverberation, AWGN) strictly on practice/dev audio and extracts the augmentation drift subspace $U_{\text{aug}}$.
2. **Block 2 — Removing Augmentation Sensitivity Without Drop in Performance (`SubspaceNullingSurgery` & `GEVDProjectionSurgery`)**: Applies $90^\circ$ Gram-Schmidt / QR task-orthogonal projection ($0.0\%$ task signal loss) and solves the Generalized Eigenvalue Problem ($\Sigma_{\text{clean}} v_k = \lambda_k \Sigma_{\text{noise}} v_k$) to filter residuals onto high-SNR channels ($r = 96$).
3. **Block 3 — Increasing Context Across Layers (`MultiTapTemporalLifter` & `lift_rectified_quadratic_rkhs`)**: Wires earlier-layer skip connections (`L8`, `L16`), $K$-tap adjacent-frame differences $[h_{t-1}, h_t, h_{t+1}]$ (velocity $\Delta_1$ and glitch curvature $\Delta_2$), and one-sided squared anomaly features $\log(1 + \max(0, z \cdot \text{sign}(\Delta\mu))^2)$.
4. **Block 4 — Finding the Optimal Weights in One Forward Pass (`OneForwardPassOptimizer` / `OFP`)**: Solves globally for readout and bridge weights in `torch.inference_mode()` (`1` forward pass, `0` backward passes) using boundary-precision sample weighting $\Pi$, Dual-Space Woodbury inversion, and exact augmentation-nulling constraints ($C_{\text{aug}}^T W = 0$).
5. **Block 5 — Compressing Audio Model Weights (`OTVelocityCompressor` / `OT-SVD`)**: Compresses linear and Transformer layers in the joint static + transition-velocity metric $\Sigma_{\text{joint}} = \text{Cov}(Z_t) + 2\,\text{Cov}(Z_t - Z_{t-1})$ with Optimal-Transport marginal variance restoration (`318M -> 86.8M` deepfake detector; `315M -> 142M` `Wav2Vec2-Large` ASR).

---

## Reproducing All Figures & Benchmarks

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Reproduce All Blogpost Figures (Figures 1–7)

```bash
# Reproduce all 7 figures + save exact numerical verification JSON
python3 scripts/reproduce_all_figures.py --figure all --out-dir results/reproduced_figures

# Or reproduce an individual figure (e.g., Figure 2: 90-degree perpendicular surgery vs. naive removal)
python3 scripts/reproduce_all_figures.py --figure 2
```

### 3. Run the Unified AudioWSP + OFP Pipeline CLI

```bash
# Run the complete 5-block pipeline across both Audio Deepfake Detection and ASR
python3 -m harness run-wsp --task both
```

### 4. Run the Unit & Evaluation-Firewall Test Suite

```bash
pytest -q
```

### 5. Reproduce Individual Model Benchmarks

```bash
# Block 4: 1-Pass OFP vs. 25-Epoch SGD/AdamW on 7,359 unseen deepfake & codec trials
python3 benchmarks/benchmark_fpo1_vs_backprop.py

# Block 2 & Block 5: GEVD channel filtering & OT-SVD 315M -> 142M compression on Wav2Vec2-Large ASR
python3 benchmarks/eval_locked_test_w2v2_gevd_and_compression.py

# Block 4 & Block 5: 32.4B Nemotron Speech-LLM OFP + CTC velocity conditioning on locked test suite
python3 benchmarks/eval_locked_test_nemotron_ofp.py
python3 benchmarks/eval_locked_test_nemotron_ctc_and_otvel.py

# Standalone 14-dataset Hugging Face Open ASR Leaderboard evaluator
python3 benchmarks/verify_standalone_ckpt_14datasets.py

# Standalone evaluator for ishiki-labs/ishiki-deepfake-0.5 (86.8M compressed & 318M full checkpoints)
python3 benchmarks/deepfake_inference.py --help
```

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `harness/audiowsp.py` | Unified implementation of all 5 Weight-Space Programming building blocks (`AcousticNuisanceProfiler`, `SubspaceNullingSurgery`, `GEVDProjectionSurgery`, `MultiTapTemporalLifter`, `OneForwardPassOptimizer`, `OTVelocityCompressor`, `AudioWSPPipeline`) |
| `harness/one_pass_frontier_optimizer.py` | Closed-form 1-Forward-Pass Optimizer (`OFP`), Dual-Space Woodbury solver, Capon-LCMV constraint projection, and `OT-SVD` solver |
| `harness/firewall.py` | Strict `EvalFirewallViolation` guard blocking any test-split file paths or identifiers during calibration |
| `harness/sandbox.py` | Process-isolated sandbox runner for LLM-generated weight edits |
| `scripts/reproduce_all_figures.py` | End-to-end reproduction script for **Figures 1–7** in the blogpost |
| `scripts/generate_visual_first_principles_figures.py` | Matplotlib static & animated figure generator |
| `benchmarks/` | Full benchmark scripts and locked-test JSON summaries (`full_benchmark_summary.json`, `w2v2_locked_test_full_and_compression.json`, `nemotron_locked_test_deep_stack_results.json`, `open_asr_leaderboard_14datasets_audiowsp_v8.json`) |
| `tests/` | Unit tests for the `AudioWSP` primitives, `OFP` solver, `OT-SVD` compressor, and evaluation firewall |

---

## Citation

```bibtex
@article{yadav2026audiowsp,
  title   = {Weight-Space Programming for Audio Models: LLMs Can Directly Edit Audio Model Weights and Compress Them},
  author  = {Yadav, Amit Kumar Singh and Xu, Robert},
  journal = {Ishiki Labs Research},
  year    = {2026},
  url     = {https://aksy1999.github.io/blog/weight-space-programming-for-audio-models/}
}
```
