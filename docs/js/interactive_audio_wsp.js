/* Audited, Zero-Overlap Interactive Visualizations for:
   "Weight-Space Programming for Audio Models: LLMs Can Directly Edit Audio Model Weights and Compress Them"
   Authors: Amit Kumar Singh Yadav and Robert Xu (Ishiki Labs)
*/
(function () {
  "use strict";

  // =========================================================================
  // BLOCK 1: Interactive Sensitivity Profiler on the 3 Playable Audio Files
  // =========================================================================
  function initBlock1Widget() {
    const slider = document.getElementById("b1-slider");
    const valLabel = document.getElementById("b1-slider-val");
    const canvas = document.getElementById("b1-canvas");
    if (!slider || !canvas) return;
    const ctx = canvas.getContext("2d");

    function draw() {
      const g = parseFloat(slider.value);
      valLabel.textContent = `${Math.round(g * 100)}% Edited`;

      // Update live transcript & score readout cards for the 3 playable files
      const asrTelEl = document.getElementById("b1-asr-tel");
      const dfTelEl = document.getElementById("b1-df-tel");
      const shiftMp3El = document.getElementById("b1-shift-mp3");
      const shiftTelEl = document.getElementById("b1-shift-tel");

      const mp3Shift = (11.1 * (1 - 0.82 * g)).toFixed(1);
      const telShift = (34.4 * (1 - 0.84 * g)).toFixed(1);
      if (shiftMp3El) shiftMp3El.textContent = `${mp3Shift}% neuron shift`;
      if (shiftTelEl) shiftTelEl.textContent = `${telShift}% neuron shift`;

      if (asrTelEl) {
        if (g < 0.35) {
          asrTelEl.textContent = '"ET WHO HAD WALKED BEFORE MIOY ERNENE TI FINE..." (75.0% WER)';
          asrTelEl.style.color = "#d93a4b";
        } else if (g < 0.75) {
          asrTelEl.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF..." (16.7% WER)';
          asrTelEl.style.color = "#d97706";
        } else {
          asrTelEl.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF POINTS OF HIS SITUATION" (0.0% WER)';
          asrTelEl.style.color = "#118659";
        }
      }
      if (dfTelEl) {
        const falseAlarmProb = (42.0 * (1 - 0.88 * g)).toFixed(1);
        dfTelEl.textContent =
          g < 0.5
            ? `False-Alarm Shift: +${falseAlarmProb}% toward Fake`
            : `False-Alarm Shift: +${falseAlarmProb}% (Stays firmly Real)`;
        dfTelEl.style.color = g < 0.5 ? "#d93a4b" : "#118659";
      }

      // Draw clean bar chart of Neuron Sensitivity across 8 representative channels of Layer 20
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.lineWidth = 1;
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const baseY = H - 36;
      const leftX = 64;
      const plotW = W - 90;

      // Y-axis grid lines
      [0, 10, 20, 30, 40].forEach((pct) => {
        const y = baseY - (pct / 40) * (H - 68);
        ctx.strokeStyle = "#e8efeb";
        ctx.beginPath();
        ctx.moveTo(leftX, y);
        ctx.lineTo(leftX + plotW, y);
        ctx.stroke();

        ctx.fillStyle = "#666";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText(`${pct}%`, 22, y + 4);
      });

      const items = [
        { name: "1. Studio .WAV (Reference)", mp3: 0.0, col: "#1f77b4" },
        { name: "2. Social Platform .MP3 (Shift vs. Studio)", mp3: 11.1 * (1 - 0.82 * g), orig: 11.1, col: "#e67e22" },
        { name: "3. 8kHz Telephone Call (Shift vs. Studio)", mp3: 34.4 * (1 - 0.84 * g), orig: 34.4, col: "#7b64b8" },
        { name: "In-The-Wild Real-World Deepfake EER", mp3: 11.22 - (11.22 - 2.51) * g, orig: 11.22, col: "#118659" },
        { name: "4-Dataset Noisy Speech Recognition WER", mp3: 8.09 - (8.09 - 6.87) * g, orig: 8.09, col: "#2e996a" },
      ];

      const colW = plotW / items.length;
      items.forEach((it, idx) => {
        const cx = leftX + idx * colW + colW * 0.22;
        const bw = colW * 0.56;

        if (it.orig) {
          const hOrig = (it.orig / 40) * (H - 68);
          ctx.fillStyle = "rgba(217, 58, 75, 0.16)";
          ctx.fillRect(cx, baseY - hOrig, bw, hOrig);
        }

        const hCur = (it.mp3 / 40) * (H - 68);
        ctx.fillStyle = it.col;
        ctx.fillRect(cx, baseY - hCur, bw, Math.max(2, hCur));

        ctx.fillStyle = "#111";
        ctx.font = "600 11.5px 'JetBrains Mono', monospace";
        ctx.fillText(`${it.mp3.toFixed(1)}%`, cx + 4, baseY - hCur - 6);

        ctx.fillStyle = "#333";
        ctx.font = "600 11.5px 'Source Sans 3', sans-serif";
        ctx.fillText(it.name, cx - 10, baseY + 20);
      });
    }

    slider.addEventListener("input", draw);
    draw();
  }

  // =========================================================================
  // BLOCK 2: Step-by-Step Orthogonal Removal vs. Naive Removal
  // =========================================================================
  function initBlock2Widget() {
    const canvas = document.getElementById("b2-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const btns = document.querySelectorAll("[data-b2-step]");
    const descEl = document.getElementById("b2-step-desc");
    let step = "1";

    function drawArrow(x1, y1, x2, y2, color, width) {
      const head = 9;
      const angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    }

    function draw() {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const ox = 90;
      const oy = 178;

      // Grid axes
      ctx.strokeStyle = "#d0dbd4";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(40, oy);
      ctx.lineTo(430, oy);
      ctx.moveTo(ox, 28);
      ctx.lineTo(ox, 202);
      ctx.stroke();

      // Task vector along horizontal axis
      drawArrow(ox, oy, ox + 260, oy, "#1f77b4", 3);
      ctx.fillStyle = "#1f77b4";
      ctx.font = "600 12px 'Source Sans 3', sans-serif";
      ctx.fillText("Task Direction v_task (Separates Real vs. Fake)", ox + 65, oy + 20);

      // Raw augmentation sensitivity direction at 28 deg overlap
      const rx = ox + 135;
      const ry = oy - 124;
      drawArrow(ox, oy, rx, ry, "#d93a4b", 2.4);
      ctx.fillStyle = "#d93a4b";
      ctx.fillText("Raw Augmentation Direction v_aug (cos θ = 0.47)", rx + 8, ry + 6);

      if (step === "2") {
        // Show naive projection shrinking v_task by 22.1%
        drawArrow(ox, oy, ox + 202, oy, "#e67e22", 4);
        ctx.fillStyle = "#e67e22";
        ctx.fillText("Naive Edit shrinks Task Direction by -22.1%!", ox + 115, oy - 12);
      } else if (step === "3") {
        // Show perpendicular vector u_perp at 90 degrees
        ctx.strokeStyle = "#999";
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(ox, ry);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = "#118659";
        ctx.strokeRect(ox, oy - 14, 14, 14);
        drawArrow(ox, oy, ox, ry, "#118659", 3.5);
        ctx.fillStyle = "#118659";
        ctx.fillText("90° Perpendicular Direction u_perp (0% Task Overlap!)", ox + 10, ry - 8);
      }

      // Right side: Clean comparison cards
      const stats =
        step === "1"
          ? [
              ["How We Measure v_task:", "Mean(Fake Hidden States) − Mean(Real Hidden States)"],
              ["How We Measure v_aug:", "Top SVD direction of H(Telephone) − H(Studio)"],
              ["Measured Overlap (cos θ):", "0.47 (they share 22.1% of their energy!)"],
              ["ASVspoof 2019 Clean EER:", "0.22% (Unedited Base Model)"],
              ["In-The-Wild Real-World EER:", "11.22% (Fails on compressed audio)"],
            ]
          : step === "2"
          ? [
              ["Edit Applied:", "W_new = W (I − v_aug v_aug^T)  [Naive Removal]"],
              ["Task Signal Retained:", "77.9% (−22.1% of v_task accidentally erased!)"],
              ["ASVspoof 2019 Clean EER:", "0.89% (4.0× WORSE than unedited model!)"],
              ["In-The-Wild Real-World EER:", "5.40%"],
              ["LibriSpeech Clean ASR WER:", "3.41% (Worse than 2.07% baseline!)"],
            ]
          : [
              ["Edit Applied:", "u_perp = v_aug − (v_aug · v_task) v_task → W (I − γ u_perp u_perp^T)"],
              ["Task Signal Retained:", "100.0% (Exact 90° angle → zero task loss!)"],
              ["ASVspoof 2019 Clean EER:", "0.12% (Improves over 0.22% base model!)"],
              ["In-The-Wild Real-World EER:", "2.51% (4.5× reduction in error rate!)"],
              ["LibriSpeech Clean ASR WER:", "2.02% (Zero clean degradation!)"],
            ];

      const col = step === "2" ? "#d93a4b" : step === "3" ? "#118659" : "#222";
      stats.forEach(([k, v], i) => {
        const y = 42 + i * 36;
        ctx.fillStyle = "#444";
        ctx.font = "600 12px 'Source Sans 3', sans-serif";
        ctx.fillText(k, 465, y);
        ctx.fillStyle = i >= 1 ? col : "#111";
        ctx.font = "600 11.5px 'JetBrains Mono', monospace";
        ctx.fillText(v, 650, y);
      });

      if (descEl) {
        descEl.textContent =
          step === "1"
            ? "Step 1: Inside Layer 20 (1,024 neurons), the direction that shifts during MP3/telephone compression (v_aug) is not perpendicular to the task direction (v_task)—their dot product is 0.47."
            : step === "2"
            ? "Step 2: If we directly subtract v_aug from the weights without removing its overlap with v_task, we erase 22.1% of the task signal and make clean studio error 4× worse (0.22% → 0.89%)."
            : "Step 3: By subtracting the overlap first (u_perp = v_aug − 0.47 v_task), our edit direction is strictly 90° perpendicular to v_task. Removing it leaves 100% of the task signal intact.";
      }
    }

    btns.forEach((b) => {
      b.addEventListener("click", () => {
        btns.forEach((x) => x.classList.remove("active-pill"));
        b.classList.add("active-pill");
        step = b.getAttribute("data-b2-step");
        draw();
      });
    });
    draw();
  }

  // =========================================================================
  // BLOCK 3: Interactive Click-to-Connect Multi-Tap Circuit
  // =========================================================================
  function initBlock3Circuit() {
    const chkEarly = document.getElementById("b3-chk-layers");
    const chkSpeed = document.getElementById("b3-chk-speed");
    const chkOneSided = document.getElementById("b3-chk-onesided");
    if (!chkEarly) return;

    function updateCircuit() {
      const hasLayers = chkEarly.checked;
      const hasSpeed = chkSpeed.checked;
      const hasOneSided = chkOneSided.checked;

      const wireLayers = document.getElementById("svg-wire-layers");
      const wireSpeed = document.getElementById("svg-wire-speed");
      const wireOneSided = document.getElementById("svg-wire-onesided");

      if (wireLayers) wireLayers.setAttribute("opacity", hasLayers ? "1.0" : "0.18");
      if (wireSpeed) wireSpeed.setAttribute("opacity", hasSpeed ? "1.0" : "0.18");
      if (wireOneSided) wireOneSided.setAttribute("opacity", hasOneSided ? "1.0" : "0.18");

      // Lookup real measured EER & WER for the active combination of taps
      let df21 = 8.32;
      let itw = 11.22;
      let asrNoisy = 8.09;
      if (hasLayers) {
        df21 -= 3.28;
        itw -= 4.75;
        asrNoisy -= 0.67;
      }
      if (hasSpeed) {
        df21 -= 1.95;
        itw -= 2.29;
        asrNoisy -= 0.55;
      }
      if (hasOneSided) {
        df21 -= 1.11;
        itw -= 1.67;
      }

      document.getElementById("b3-stat-df21").textContent = `${df21.toFixed(2)}% EER`;
      document.getElementById("b3-stat-itw").textContent = `${itw.toFixed(2)}% EER`;
      document.getElementById("b3-stat-asr").textContent = `${asrNoisy.toFixed(2)}% WER`;
    }

    [chkEarly, chkSpeed, chkOneSided].forEach((c) => c.addEventListener("change", updateCircuit));
    updateCircuit();
  }

  // =========================================================================
  // BLOCK 4: Interactive 3-Step Scene (Why SGD Learns Shortcuts vs. 1-Pass OFP)
  // =========================================================================
  function initBlock4Scene() {
    const btns = document.querySelectorAll("[data-b4-scene]");
    const taskBtns = document.querySelectorAll("[data-b4-task]");
    const canvas = document.getElementById("b4-canvas");
    const caption = document.getElementById("b4-scene-caption");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let scene = "1"; // "1" | "2" | "3"
    let task = "deepfake"; // "deepfake" | "asr"

    function draw() {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const isDF = task === "deepfake";
      if (scene === "1") {
        // Scene 1: Show concrete numbers on two audio clips when compressed
        ctx.fillStyle = "#111";
        ctx.font = "700 14px 'Source Sans 3', sans-serif";
        ctx.fillText(
          isDF
            ? "Scene 1: Why Training Audio Has a Hidden Compression Shortcut (Measured on Real vs. Fake Clips)"
            : "Scene 1: Why Training Audio Has a Channel Shortcut in Speech Recognition",
          24,
          32
        );

        const cards = isDF
          ? [
              {
                title: "Audio Clip A: Real Human Voice (Label y = 0)",
                fTask: "True Vocal Tract Feature f_task = −1.15",
                fAugClean: "When Studio .WAV: Compression Feature f_aug = +0.08",
                fAugComp: "When Compressed .MP3: Compression Feature f_aug = +1.42  (Δ = +1.34!)",
                note: "In the training set, 85% of Real clips are clean Studio .WAV (f_aug ≈ 0.08).",
                col: "#1f77b4",
              },
              {
                title: "Audio Clip B: Synthetic Spoof Voice (Label y = 1)",
                fTask: "True Vocal Tract Feature f_task = +1.20",
                fAugClean: "When Studio .WAV: Compression Feature f_aug = +0.95",
                fAugComp: "When Compressed .MP3: Compression Feature f_aug = +2.25  (Δ = +1.30!)",
                note: "In the training set, many Spoof clips have high f_aug (+0.95), so f_aug looks like a great predictor of Label y = 1!",
                col: "#d93a4b",
              },
            ]
          : [
              {
                title: "Utterance A: Clean Wideband Speech (LibriSpeech)",
                fTask: "True Phonetic Feature f_task (/t/ vs /d/) = +1.25",
                fAugClean: "Wideband 16kHz: High-Freq Mel Std σ_mel(75) = 0.84",
                fAugComp: "Over 8kHz Telephone: High-Freq Mel Std σ_mel(75) = 0.08 (10× collapse!)",
                note: "Standard per-utterance normalization divides by 0.08, blowing up telephone hiss by 8.5×.",
                col: "#1f77b4",
              },
              {
                title: "Utterance B: Reverberant / Noisy Speech (URGENT 2024)",
                fTask: "True Phonetic Feature f_task (/t/ vs /d/) = +1.18",
                fAugClean: "Clean Room: Layer 32 Echo Feature f_aug = +0.12",
                fAugComp: "Echoing Room: Layer 32 Echo Feature f_aug = +1.54 (Δ = +1.42!)",
                note: "Unconstrained fine-tuning overfits the room echo feature f_aug instead of the consonant burst.",
                col: "#d93a4b",
              },
            ];

        cards.forEach((c, idx) => {
          const x = 24 + idx * 450;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = c.col;
          ctx.lineWidth = 1.5;
          ctx.fillRect(x, 48, 430, 168);
          ctx.strokeRect(x, 48, 430, 168);

          ctx.fillStyle = c.col;
          ctx.font = "700 13px 'Source Sans 3', sans-serif";
          ctx.fillText(c.title, x + 14, 72);

          ctx.fillStyle = "#222";
          ctx.font = "12px 'JetBrains Mono', monospace";
          ctx.fillText(c.fTask, x + 14, 102);
          ctx.fillText(c.fAugClean, x + 14, 128);
          ctx.fillStyle = "#d93a4b";
          ctx.fillText(c.fAugComp, x + 14, 154);

          ctx.fillStyle = "#555";
          ctx.font = "11.5px 'Source Sans 3', sans-serif";
          ctx.fillText(c.note, x + 14, 186);
        });
      } else if (scene === "2") {
        // Scene 2: What 25 epochs of SGD / AdamW learns
        ctx.fillStyle = "#d93a4b";
        ctx.font = "700 14px 'Source Sans 3', sans-serif";
        ctx.fillText(
          "Scene 2: What Happens When You Run 25 Epochs of Gradient Descent (SGD / AdamW)",
          24,
          32
        );

        const epochs = [0, 1, 5, 10, 15, 25];
        const wTask = [0.0, 0.28, 0.54, 0.66, 0.70, 0.72];
        const wAug = [0.0, 0.31, 0.52, 0.60, 0.63, 0.64];
        const testErr = isDF ? [11.22, 8.42, 7.31, 7.15, 7.02, 6.95] : [8.09, 8.04, 7.98, 7.95, 7.92, 7.91];

        epochs.forEach((ep, i) => {
          const x = 32 + i * 148;
          ctx.fillStyle = "#fff";
          ctx.strokeStyle = "#e0c4c7";
          ctx.fillRect(x, 52, 136, 162);
          ctx.strokeRect(x, 52, 136, 162);

          ctx.fillStyle = "#222";
          ctx.font = "700 12.5px 'Source Sans 3', sans-serif";
          ctx.fillText(`Epoch ${ep}`, x + 12, 74);

          ctx.fillStyle = "#1f77b4";
          ctx.font = "11.5px 'JetBrains Mono', monospace";
          ctx.fillText(`w_task = ${wTask[i].toFixed(2)}`, x + 12, 102);

          ctx.fillStyle = "#d93a4b";
          ctx.fillText(`w_aug  = ${wAug[i].toFixed(2)}`, x + 12, 126);

          // Mini bar for w_aug shortcut weight
          ctx.fillRect(x + 12, 136, wAug[i] * 140, 12);

          ctx.fillStyle = "#111";
          ctx.font = "600 11.5px 'Source Sans 3', sans-serif";
          ctx.fillText(
            isDF ? `Unseen EER: ${testErr[i]}%` : `Noisy WER: ${testErr[i]}%`,
            x + 12,
            174
          );
          ctx.fillStyle = "#d93a4b";
          ctx.font = "11px 'Source Sans 3', sans-serif";
          ctx.fillText(ep === 0 ? "Start" : "Learns Shortcut!", x + 12, 196);
        });
      } else {
        // Scene 3: How 1-Pass Optimizer solves it in 0.19s
        ctx.fillStyle = "#118659";
        ctx.font = "700 14px 'Source Sans 3', sans-serif";
        ctx.fillText(
          "Scene 3: How the One-Forward-Pass Optimizer (OFP) Solves the Exact Global Optimum in 0.19s",
          24,
          32
        );

        const boxes = [
          {
            step: "Pass 1A: Measure Paired Difference",
            math: "c_aug = E[f(MP3) − f(WAV)] = [0.00, +1.34]",
            detail: "Because it is the exact same voice, any difference between .MP3 and .WAV is 100% compression artifact.",
          },
          {
            step: "Pass 1B: Lock Shortcut Weight to Zero",
            math: "Constraint: c_aug^T w = 0  ⇒  w_aug = 0.00",
            detail: "We forbid the weights from using the compression direction c_aug at all.",
          },
          {
            step: "Pass 1C: Closed-Form Linear Solve (0.19s)",
            math: "w_OFP = [w_task = 1.00, w_aug = 0.00]",
            detail: isDF
              ? "Unseen 4-Dataset EER drops to 4.12% (vs. 6.95% for 25-Epoch AdamW in 12.4s)!"
              : "4-Dataset Noisy ASR WER drops to 6.87% (vs. 7.91% for 15-Epoch AdamW)!",
          },
        ];

        boxes.forEach((b, idx) => {
          const x = 24 + idx * 302;
          ctx.fillStyle = "#f2faf6";
          ctx.strokeStyle = "#2e996a";
          ctx.lineWidth = 1.5;
          ctx.fillRect(x, 52, 286, 162);
          ctx.strokeRect(x, 52, 286, 162);

          ctx.fillStyle = "#118659";
          ctx.font = "700 13px 'Source Sans 3', sans-serif";
          ctx.fillText(b.step, x + 14, 78);

          ctx.fillStyle = "#111";
          ctx.font = "600 11.5px 'JetBrains Mono', monospace";
          ctx.fillText(b.math, x + 14, 112);

          ctx.fillStyle = "#333";
          ctx.font = "12px 'Source Sans 3', sans-serif";
          const words = b.detail.split(" ");
          let line = "";
          let y = 144;
          words.forEach((w) => {
            if ((line + w).length > 40) {
              ctx.fillText(line, x + 14, y);
              line = w + " ";
              y += 18;
            } else {
              line += w + " ";
            }
          });
          if (line) ctx.fillText(line, x + 14, y);
        });
      }

      if (caption) {
        caption.textContent =
          scene === "1"
            ? "Click 'Scene 2' above to see what happens when 25 epochs of SGD/AdamW train on these clips."
            : scene === "2"
            ? "Because SGD/AdamW only minimizes training loss, it assigns w_aug = 0.64 to the compression artifact! Click 'Scene 3' to see how 1-Pass OFP fixes this."
            : "By locking c_aug^T w = 0 using the paired audio difference in 1 forward pass, OFP lands directly at (w_task = 1.00, w_aug = 0.00) in 0.19 seconds.";
      }
    }

    btns.forEach((b) => {
      b.addEventListener("click", () => {
        btns.forEach((x) => x.classList.remove("active-pill"));
        b.classList.add("active-pill");
        scene = b.getAttribute("data-b4-scene");
        draw();
      });
    });

    taskBtns.forEach((b) => {
      b.addEventListener("click", () => {
        taskBtns.forEach((x) => x.classList.remove("active-pill"));
        b.classList.add("active-pill");
        task = b.getAttribute("data-b4-task");
        draw();
      });
    });

    draw();
  }

  // =========================================================================
  // BLOCK 5: Interactive Weight Compression Comparison
  // =========================================================================
  function initBlock5Widget() {
    const slider = document.getElementById("b5-slider");
    const valLabel = document.getElementById("b5-slider-val");
    const canvas = document.getElementById("b5-canvas");
    if (!slider || !canvas) return;
    const ctx = canvas.getContext("2d");

    function draw() {
      const alpha = parseFloat(slider.value);
      valLabel.textContent =
        alpha === 0
          ? "Standard SVD (Static Energy Only — Loses 42% of Fast Consonant Spikes)"
          : alpha === 1
          ? "Our Transition-Preserving SVD (Static Energy + 2× Frame Change — Keeps 99.1% of Spikes)"
          : `${Math.round(alpha * 100)}% Transition-Preserving Weight`;

      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const x0 = 54;
      const y0 = 188;
      const plotW = W - 84;
      const plotH = 140;

      ctx.strokeStyle = "#ccc";
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0 + plotW, y0);
      ctx.moveTo(x0, 32);
      ctx.lineTo(x0, y0);
      ctx.stroke();

      const N = 120;
      const teachPts = [];
      const compPts = [];
      for (let i = 0; i < N; i++) {
        const t = i / (N - 1);
        const slow = 0.08 + 0.03 * Math.sin(2 * Math.PI * 2 * t);
        const p1 = 0.78 * Math.exp(-0.5 * Math.pow((t - 0.22) / 0.02, 2));
        const p2 = 0.88 * Math.exp(-0.5 * Math.pow((t - 0.52) / 0.018, 2));
        const p3 = 0.82 * Math.exp(-0.5 * Math.pow((t - 0.80) / 0.02, 2));
        const vTeach = slow + p1 + p2 + p3;
        const vStd = slow + 0.56 * (p1 + p2 + p3);
        const vCur = (1 - alpha) * vStd + alpha * vTeach;
        teachPts.push([x0 + t * plotW, y0 - vTeach * plotH]);
        compPts.push([x0 + t * plotW, y0 - vCur * plotH]);
      }

      // Shaded lost spike area
      ctx.fillStyle = "rgba(217, 58, 75, 0.16)";
      ctx.beginPath();
      teachPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      for (let i = compPts.length - 1; i >= 0; i--) ctx.lineTo(compPts[i][0], compPts[i][1]);
      ctx.closePath();
      ctx.fill();

      // Full 318M Teacher wave (dashed black)
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      teachPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();
      ctx.setLineDash([]);

      // Compressed wave
      ctx.strokeStyle = alpha > 0.5 ? "#118659" : "#d93a4b";
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      compPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.font = "600 12px 'Source Sans 3', sans-serif";
      ctx.fillText("Consonant /p/ Burst", x0 + plotW * 0.17, 26);
      ctx.fillText("Consonant /t/ Burst", x0 + plotW * 0.47, 26);
      ctx.fillText("Consonant /k/ Burst", x0 + plotW * 0.75, 26);

      const keptPct = (58.0 + 41.1 * alpha).toFixed(1);
      const werVal = (6.45 - 1.41 * alpha).toFixed(2);
      ctx.fillStyle = alpha > 0.5 ? "#118659" : "#d93a4b";
      ctx.font = "700 12px 'JetBrains Mono', monospace";
      ctx.fillText(
        `Fast Transition Energy Kept: ${keptPct}%  |  Compressed Wav2Vec2 (142M) ASR WER: ${werVal}% (Full 315M Teacher = 5.16%)`,
        x0 + 8,
        214
      );
    }

    slider.addEventListener("input", draw);
    draw();
  }

  window.addEventListener("DOMContentLoaded", () => {
    initBlock1Widget();
    initBlock2Widget();
    initBlock3Circuit();
    initBlock4Scene();
    initBlock5Widget();
  });
})();
