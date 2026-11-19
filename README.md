# AudioWSP: Unified Weight-Space Programming & One-Forward-Pass Optimization Harness for Audio Models

`audiowsp` (`harness/audiowsp.py`) is the open-source Python harness accompanying **"Weight-Space Programming for Audio Models: LLMs Can Directly Edit Audio Model Weights and Compress Them"** by **Amit Kumar Singh Yadav** and **Ishiki Labs Research**.

It provides the 5 common first-principles building blocks (guarded by `firewall.py`'s `EvalFirewallViolation` anti-leakage checks) used across both **Audio Deepfake Detection (`ishiki-labs/ishiki-deepfake-0.5`, `318M -> 86.8M`)** and **Automatic Speech Recognition (`ishiki-labs/ishiki-asr-0.5`, `32.4B -> 26.2B`)**:

1. **Block 1 — Acoustic Nuisance Profiler (`AcousticNuisanceProfiler`)**: Generates frame-aligned paired clean vs. corrupted probes (`.WAV` vs. `.MP3`, `Encodec`, `8kHz` PSTN $\mu$-law, RIR reverberation, AWGN) and extracts the task-orthogonal nuisance drift subspace $U_{\text{nuis}}$.
2. **Block 2 — Task-Orthogonal Weight Surgery & Signal-to-Noise Eigen-Filtering (`SubspaceNullingSurgery` & `GEVDProjectionSurgery`)**: Applies $90^\circ$ Gram-Schmidt / QR task-orthogonal null-space projection ($0.0\%$ task signal loss) and solves the Generalized Eigenvalue Problem ($\Sigma_{\text{clean}} v_k = \lambda_k \Sigma_{\text{nuis}} v_k$) to filter residuals onto high-SNR channels ($r = 96$).
3. **Block 3 — Multi-Tap Temporal Derivatives & One-Sided Squared Anomaly Scoring (`MultiTapTemporalLifter` & `lift_rectified_quadratic_rkhs`)**: Wires $K$-tap temporal windows $[h_{t-1}, h_t, h_{t+1}]$ (velocity $\Delta_1$ and glitch curvature $\Delta_2$) and one-sided rectified quadratic features $\log(1 + \max(0, z \cdot \text{sign}(\Delta\mu))^2)$.
4. **Block 4 — One-Forward-Pass Optimizer (`OneForwardPassOptimizer` & `solve_ofp_ctc_readout`)**: Solves for optimal readout/bridge weights in `torch.inference_mode()` (`1` forward pass, `0` backward passes) using Boltzmann boundary precision re-weighting, Dual-Space Woodbury inversion, and Capon-LCMV nuisance nulling ($C_{\text{codec}}^T W = 0$).
5. **Block 5 — Velocity-Preserving Compression (`OTVelocityCompressor` & `MoEReLU2Compactor`)**: Compresses linear and Transformer layers in the joint static + velocity metric $\Sigma_{\text{joint}} = \text{Cov}(Z_t) + 2\,\text{Cov}(Z_t - Z_{t-1})$ with Optimal-Transport marginal variance restoration.

## Quickstart CLI

```bash
# Run the unified 6-block AudioWSP + OFP pipeline on both Deepfake Detection and ASR
python -m harness run-wsp --task both
```
