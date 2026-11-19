/* Clean, Zero-Overlap Visual Animations for:
   "Weight-Space Programming for Audio Models: LLMs Can Directly Edit Audio Model Weights and Compress Them"
   Authors: Amit Kumar Singh Yadav and Robert Xu (Ishiki Labs)
*/
(function () {
  "use strict";

  // =========================================================================
  // BLOCK 1: Separate ASR (WER %) & Deepfake (EER %) Charts + Live Transcript
  // =========================================================================
  function initBlock1Widget() {
    const slider = document.getElementById("b1-slider");
    const valLabel = document.getElementById("b1-slider-val");
    const canvas = document.getElementById("b1-canvas");
    if (!slider || !canvas) return;
    const ctx = canvas.getContext("2d");

    function drawBarPanel(x0, y0, w, h, title, yMax, yUnit, bars) {
      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.lineWidth = 1;
      ctx.fillRect(x0, y0, w, h);
      ctx.strokeRect(x0, y0, w, h);

      ctx.fillStyle = "#1b3b2b";
      ctx.font = "700 13px 'Source Sans 3', sans-serif";
      ctx.fillText(title, x0 + 14, y0 + 24);

      const plotLeft = x0 + 48;
      const plotBottom = y0 + h - 34;
      const plotTop = y0 + 44;
      const plotH = plotBottom - plotTop;
      const plotW = w - 64;

      // Grid lines
      for (let i = 0; i <= 4; i++) {
        const val = (yMax * i) / 4;
        const py = plotBottom - (val / yMax) * plotH;
        ctx.strokeStyle = "#e6efea";
        ctx.beginPath();
        ctx.moveTo(plotLeft, py);
        ctx.lineTo(plotLeft + plotW, py);
        ctx.stroke();

        ctx.fillStyle = "#666";
        ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText(`${val.toFixed(0)}${yUnit}`, x0 + 10, py + 4);
      }

      const slotW = plotW / bars.length;
      bars.forEach((b, idx) => {
        const bw = Math.min(48, slotW * 0.54);
        const bx = plotLeft + idx * slotW + (slotW - bw) / 2;

        // Ghost bar showing unedited 0% baseline
        const hOrig = (b.orig / yMax) * plotH;
        ctx.fillStyle = "rgba(217, 58, 75, 0.16)";
        ctx.fillRect(bx, plotBottom - hOrig, bw, hOrig);

        // Current bar
        const hCur = (b.cur / yMax) * plotH;
        ctx.fillStyle = b.color;
        ctx.fillRect(bx, plotBottom - hCur, bw, Math.max(2, hCur));

        // Value above bar
        ctx.fillStyle = "#111";
        ctx.font = "700 11.5px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${b.cur.toFixed(1)}${yUnit}`, bx + bw / 2, plotBottom - hCur - 6);

        // Short X-axis label (never overlaps!)
        ctx.fillStyle = "#333";
        ctx.font = "600 11.5px 'Source Sans 3', sans-serif";
        ctx.fillText(b.label, bx + bw / 2, plotBottom + 18);
        ctx.textAlign = "left";
      });
    }

    function draw() {
      const g = parseFloat(slider.value);
      valLabel.textContent =
        g === 0
          ? "0% (Unedited Base Model W_orig)"
          : g === 1
          ? "100% (Fully Edited WSP Weights W_WSP)"
          : `${Math.round(g * 100)}% Edited (W = (1−γ)W_orig + γW_WSP)`;

      const asrMp3El = document.getElementById("b1-asr-mp3");
      const mp3BadgeEl = document.getElementById("b1-mp3-wer-badge");
      if (asrMp3El && mp3BadgeEl) {
        if (g < 0.35) {
          asrMp3El.textContent = '"HURSHWOOD WALKED BEFORE MENTALLY ERRING TING THE DEAP POINT OF TE TI PELATION"';
          asrMp3El.style.color = "#d93a4b";
          mp3BadgeEl.textContent = "66.7% WER (8 / 12 words wrong)";
          mp3BadgeEl.style.color = "#d93a4b";
        } else if (g < 0.75) {
          asrMp3El.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE DEAP POINTS OF HIS SITUATION"';
          asrMp3El.style.color = "#d97706";
          mp3BadgeEl.textContent = "8.3% WER (1 / 12 words wrong)";
          mp3BadgeEl.style.color = "#d97706";
        } else {
          asrMp3El.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF POINTS OF HIS SITUATION"';
          asrMp3El.style.color = "#118659";
          mp3BadgeEl.textContent = "0.0% WER (All 12 words correct!)";
          mp3BadgeEl.style.color = "#118659";
        }
      }

      const asrTelEl = document.getElementById("b1-asr-tel");
      const werBadgeEl = document.getElementById("b1-tel-wer-badge");
      if (asrTelEl && werBadgeEl) {
        if (g < 0.35) {
          asrTelEl.textContent = '"ET WHO HAD WALKED BEFORE MIOY ERNENE TI FINE TE TE TEDERD"';
          asrTelEl.style.color = "#d93a4b";
          werBadgeEl.textContent = "75.0% WER (9 / 12 words wrong)";
          werBadgeEl.style.color = "#d93a4b";
        } else if (g < 0.75) {
          asrTelEl.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF..."';
          asrTelEl.style.color = "#d97706";
          werBadgeEl.textContent = "16.7% WER (2 / 12 words wrong)";
          werBadgeEl.style.color = "#d97706";
        } else {
          asrTelEl.textContent = '"HURSTWOOD WALKED THE FLOOR MENTALLY ARRANGING THE CHIEF POINTS OF HIS SITUATION"';
          asrTelEl.style.color = "#118659";
          werBadgeEl.textContent = "0.0% WER (All 12 words correct!)";
          werBadgeEl.style.color = "#118659";
        }
      }

      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const panelW = (W - 20) / 2;
      drawBarPanel(
        0,
        0,
        panelW,
        H,
        "A. Speech Recognition Word Error Rate (WER %, Lower is Better)",
        80,
        "%",
        [
          { label: "Studio .WAV", orig: 0.0, cur: 0.0, color: "#1f77b4" },
          { label: "Social .MP3", orig: 66.7, cur: 66.7 * (1 - g), color: "#e67e22" },
          { label: "8kHz Phone Clip", orig: 75.0, cur: 75.0 * (1 - g), color: "#d93a4b" },
          { label: "4-Set Noisy ASR", orig: 8.09, cur: 8.09 - (8.09 - 6.87) * g, color: "#118659" },
        ]
      );

      drawBarPanel(
        panelW + 20,
        0,
        panelW,
        H,
        "B. Deepfake Detection Equal Error Rate (EER %, Lower is Better)",
        16,
        "%",
        [
          { label: "Studio (ASV19)", orig: 0.22, cur: 0.22 - (0.22 - 0.12) * g, color: "#1f77b4" },
          { label: "Codec (ASV21 DF)", orig: 8.32, cur: 8.32 - (8.32 - 1.98) * g, color: "#e67e22" },
          { label: "Phone (ASV21 LA)", orig: 8.11, cur: 8.11 - (8.11 - 1.42) * g, color: "#7b64b8" },
          { label: "In-The-Wild", orig: 11.22, cur: 11.22 - (11.22 - 2.51) * g, color: "#118659" },
        ]
      );
    }

    slider.addEventListener("input", draw);
    draw();
  }

  // =========================================================================
  // BLOCK 2A: Clean 2D Vector Diagram + Unclipped HTML Metrics Table
  // =========================================================================
  function initBlock2AWidget() {
    const canvas = document.getElementById("b2a-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const btns = document.querySelectorAll("[data-b2-step]");
    let step = "3";

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

      const ox = 56;
      const oy = 172;

      ctx.strokeStyle = "#d0dbd4";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(24, oy);
      ctx.lineTo(W - 24, oy);
      ctx.moveTo(ox, 22);
      ctx.lineTo(ox, H - 18);
      ctx.stroke();

      // Task direction v_task
      drawArrow(ox, oy, ox + 250, oy, "#1f77b4", 3);
      ctx.fillStyle = "#1f77b4";
      ctx.font = "700 12px 'Source Sans 3', sans-serif";
      ctx.fillText("Task Direction v_task", ox + 75, oy + 20);

      // Raw augmentation direction v_aug
      const rx = ox + 128;
      const ry = oy - 118;
      drawArrow(ox, oy, rx, ry, "#d93a4b", 2.4);
      ctx.fillStyle = "#d93a4b";
      ctx.fillText("Raw Augmentation v_aug (dot product = 0.47)", rx + 8, ry + 8);

      if (step === "2") {
        drawArrow(ox, oy, ox + 195, oy, "#e67e22", 4);
        ctx.fillStyle = "#e67e22";
        ctx.fillText("Naive Removal cuts −22.1% of v_task!", ox + 90, oy - 12);
      } else if (step === "3") {
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
        ctx.fillText("90° Perpendicular u_perp (0% Overlap)", ox + 10, ry - 8);
      }

      const data =
        step === "1"
          ? {
              eq: "Unedited Weights W",
              ret: "100.0%",
              asv19: "0.22% EER",
              itw: "11.22% EER (Sensitive to MP3/Phone)",
              asr: "2.07% WER",
              col: "#333",
            }
          : step === "2"
          ? {
              eq: "W (I − v_aug v_aug^T)  [Naive Removal]",
              ret: "77.9% (−22.1% of Task Erased!)",
              asv19: "0.89% EER (4.0× Worse on Studio!)",
              itw: "5.40% EER",
              asr: "3.41% WER (Degrades Clean Speech!)",
              col: "#d93a4b",
            }
          : {
              eq: "W (I − γ u_perp u_perp^T)  [90° Perpendicular]",
              ret: "100.0% (Zero Task Loss!)",
              asv19: "0.12% EER (Better than Base!)",
              itw: "2.51% EER (4.5× Lower Error!)",
              asr: "2.02% WER (Zero Clean Drop!)",
              col: "#118659",
            };

      document.getElementById("b2-eq").textContent = data.eq;
      document.getElementById("b2-ret").textContent = data.ret;
      document.getElementById("b2-asv19").textContent = data.asv19;
      document.getElementById("b2-itw").textContent = data.itw;
      document.getElementById("b2-asr").textContent = data.asr;
      ["b2-ret", "b2-asv19", "b2-itw", "b2-asr"].forEach((id) => {
        document.getElementById(id).style.color = data.col;
      });
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
  // BLOCK 2B: Interactive Signal-vs-Noise Channel Filter (GEVD Visual)
  // =========================================================================
  function initBlock2BWidget() {
    const slider = document.getElementById("b2b-slider");
    const valLabel = document.getElementById("b2b-val");
    const canvas = document.getElementById("b2b-canvas");
    if (!slider || !canvas) return;
    const ctx = canvas.getContext("2d");

    const ranks = [32, 64, 96, 128, 256, 1280];
    const wers = { 32: 7.18, 64: 6.93, 96: 6.87, 128: 7.01, 256: 7.34, 1280: 7.81 };
    const buckets = [
      { k: "1..16", sig: 88, nois: 5, maxR: 16 },
      { k: "17..32", sig: 74, nois: 8, maxR: 32 },
      { k: "33..48", sig: 60, nois: 11, maxR: 48 },
      { k: "49..64", sig: 46, nois: 14, maxR: 64 },
      { k: "65..80", sig: 34, nois: 18, maxR: 80 },
      { k: "81..96", sig: 25, nois: 22, maxR: 96 },
      { k: "97..128", sig: 14, nois: 30, maxR: 128 },
      { k: "129..256", sig: 8, nois: 38, maxR: 256 },
      { k: "257..512", sig: 4, nois: 45, maxR: 512 },
      { k: "513..1280", sig: 2, nois: 52, maxR: 1280 },
    ];

    function draw() {
      const r = ranks[parseInt(slider.value, 10)] || 96;
      const wer = wers[r];
      valLabel.textContent =
        r === 96
          ? `Keep Top 96 Channels (Speech > Noise Cutoff) → 6.87% Noisy WER (Best!)`
          : r === 1280
          ? `Keep All 1,280 Channels (Includes 1,184 Noisy Channels!) → 7.81% WER`
          : `Keep Top ${r} Channels → ${wer.toFixed(2)}% Noisy WER`;

      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const baseY = H - 36;
      const x0 = 48;
      const stepX = (W - 76) / buckets.length;

      buckets.forEach((b, i) => {
        const bx = x0 + i * stepX;
        const kept = b.maxR <= r;
        const alpha = kept ? 1.0 : 0.2;

        // Clean Speech bar (Green)
        ctx.fillStyle = `rgba(17, 134, 89, ${alpha})`;
        const hSig = b.sig * 1.45;
        ctx.fillRect(bx, baseY - hSig, 22, hSig);

        // Background Noise bar (Red)
        ctx.fillStyle = `rgba(217, 58, 75, ${alpha})`;
        const hNois = b.nois * 1.45;
        ctx.fillRect(bx + 24, baseY - hNois, 22, hNois);

        ctx.fillStyle = kept ? "#222" : "#999";
        ctx.font = "600 11px 'JetBrains Mono', monospace";
        ctx.fillText(`Ch ${b.k}`, bx - 2, baseY + 18);
      });

      // Cutoff marker line after 81..96
      const cutX = x0 + 6 * stepX - 10;
      ctx.strokeStyle = "#118659";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(cutX, 18);
      ctx.lineTo(cutX, baseY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#118659";
      ctx.font = "700 12px 'Source Sans 3', sans-serif";
      ctx.fillText("← Top 96 Channels: Speech (Green) > Noise (Red)", cutX - 275, 28);
      ctx.fillStyle = "#d93a4b";
      ctx.fillText("Channels 97..1280: Noise (Red) > Speech (Green) →", cutX + 10, 28);
    }

    slider.addEventListener("input", draw);
    draw();
  }

  // =========================================================================
  // BLOCK 3: Layer-by-Layer Proof Chart + Numbered 1, 2, 3 Multi-Tap Circuit
  // =========================================================================
  function initBlock3Circuit() {
    // Draw the Layer-by-Layer Proof Chart showing why Layer 24 alone loses early pitch/phase info
    const layerCanvas = document.getElementById("b3-layer-canvas");
    if (layerCanvas) {
      const lctx = layerCanvas.getContext("2d");
      const W = layerCanvas.width;
      const H = layerCanvas.height;
      lctx.fillStyle = "#fafcfb";
      lctx.strokeStyle = "#d5e0da";
      lctx.fillRect(0, 0, W, H);
      lctx.strokeRect(0, 0, W, H);

      lctx.fillStyle = "#1b3b2b";
      lctx.font = "700 13px 'Source Sans 3', sans-serif";
      lctx.fillText(
        "Measured Proof: Where Vocoder Pitch/Phase Artifacts vs. Phonetic Context Peak Across Layers 1–24",
        18,
        24
      );

      const x0 = 56;
      const y0 = H - 34;
      const pw = W - 84;
      const ph = H - 74;

      lctx.strokeStyle = "#ccc";
      lctx.beginPath();
      lctx.moveTo(x0, y0);
      lctx.lineTo(x0 + pw, y0);
      lctx.moveTo(x0, 38);
      lctx.lineTo(x0, y0);
      lctx.stroke();

      // Curve 1: Fine-Grained Pitch & Vocoder Phase Sensitivity (peaks in Layers 6..14, drops by Layer 24!)
      const pitchCurve = [
        0.42, 0.58, 0.74, 0.86, 0.94, 0.96, 0.91, 0.84, 0.72, 0.58, 0.45, 0.34,
      ];
      // Curve 2: High-Level Word / Phonetic Context (peaks in Layers 18..24)
      const wordCurve = [
        0.22, 0.28, 0.35, 0.44, 0.54, 0.65, 0.75, 0.84, 0.91, 0.95, 0.97, 0.96,
      ];
      const tapLabels = ["L0", "L2", "L5", "L8", "L10", "L12", "L14", "L16", "L18", "L20", "L22", "L24"];

      function plotCurve(arr, col) {
        lctx.strokeStyle = col;
        lctx.lineWidth = 2.6;
        lctx.beginPath();
        arr.forEach((v, i) => {
          const px = x0 + (i / (arr.length - 1)) * pw;
          const py = y0 - v * ph;
          if (i === 0) lctx.moveTo(px, py);
          else lctx.lineTo(px, py);
        });
        lctx.stroke();
        arr.forEach((v, i) => {
          const px = x0 + (i / (arr.length - 1)) * pw;
          const py = y0 - v * ph;
          lctx.fillStyle = col;
          lctx.beginPath();
          lctx.arc(px, py, 3.8, 0, Math.PI * 2);
          lctx.fill();
        });
      }

      plotCurve(pitchCurve, "#118659");
      plotCurve(wordCurve, "#1f77b4");

      tapLabels.forEach((lb, i) => {
        const px = x0 + (i / (tapLabels.length - 1)) * pw;
        lctx.fillStyle = "#333";
        lctx.font = "600 11px 'JetBrains Mono', monospace";
        lctx.fillText(lb, px - 8, y0 + 18);
      });

      lctx.fillStyle = "#118659";
      lctx.font = "700 12px 'Source Sans 3', sans-serif";
      lctx.fillText("● Green: Vocoder Pitch/Phase Signal (Peaks at Layers 8–12, smoothed away by Layer 24!)", x0 + 16, 52);
      lctx.fillStyle = "#1f77b4";
      lctx.fillText("● Blue: Word / Phonetic Context (Peaks at Layers 20–24)", x0 + 540, 52);
    }

    const chkEarly = document.getElementById("b3-chk-layers");
    const chkSpeed = document.getElementById("b3-chk-speed");
    const chkOneSided = document.getElementById("b3-chk-onesided");
    if (!chkEarly) return;

    function updateCircuit() {
      const hasLayers = chkEarly.checked;
      const hasSpeed = chkSpeed.checked;
      const hasOneSided = chkOneSided.checked;

      document.getElementById("svg-wire-layers").setAttribute("opacity", hasLayers ? "1.0" : "0.15");
      document.getElementById("svg-wire-speed").setAttribute("opacity", hasSpeed ? "1.0" : "0.15");
      document.getElementById("svg-wire-onesided").setAttribute("opacity", hasOneSided ? "1.0" : "0.15");

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
  // BLOCK 4: Visual 2D Feature-Space Plot Showing Why SGD Learns the Shortcut
  // =========================================================================
  function initBlock4Scene() {
    const btns = document.querySelectorAll("[data-b4-scene]");
    const canvas = document.getElementById("b4-canvas");
    const caption = document.getElementById("b4-scene-caption");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let scene = "1"; // "1" | "2" | "3"

    // Training Real (Bottom-Left: Clean WAV), Training Fake (Top-Right: Compressed), Unseen Test Real on Phone/MP3 (Top-Left!)
    const trainReal = [
      [-1.3, -0.9], [-1.1, -0.7], [-1.4, -0.6], [-0.9, -0.8], [-1.2, -1.0], [-1.0, -0.5],
    ];
    const trainFake = [
      [1.1, 0.8], [1.3, 1.0], [0.9, 0.9], [1.4, 0.6], [1.2, 1.1], [1.0, 0.7],
    ];
    const testRealMP3 = [
      [-1.25, 0.85], [-1.05, 1.05], [-1.35, 0.95], [-0.95, 0.78], [-1.15, 1.12],
    ];

    function draw() {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Left Panel: 2D Feature Space (X = True Task Signal, Y = Compression Artifact Level)
      ctx.fillStyle = "#fafcfb";
      ctx.strokeStyle = "#d5e0da";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeRect(0, 0, W, H);

      const ox = 235;
      const oy = 135;
      const sx = 88;
      const sy = 64;

      // Shaded regions for decision boundary
      if (scene === "2") {
        // SGD Tilted Boundary: y = -1.1 * x (Misclassifies Top-Left Test Real MP3 as Fake!)
        ctx.fillStyle = "rgba(217, 58, 75, 0.08)";
        ctx.beginPath();
        ctx.moveTo(ox - 160, oy - 110);
        ctx.lineTo(ox + 180, oy - 110);
        ctx.lineTo(ox + 180, oy + 100);
        ctx.lineTo(ox + 90, oy + 100);
        ctx.closePath();
        ctx.fill();
      } else if (scene === "3") {
        // 1-Pass OFP Vertical Boundary: x = 0 (Zero weight on Y-axis!)
        ctx.fillStyle = "rgba(17, 134, 89, 0.07)";
        ctx.fillRect(20, 20, ox - 20, H - 44);
      }

      // Axes
      ctx.strokeStyle = "#c4d1c9";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(24, oy);
      ctx.lineTo(445, oy);
      ctx.moveTo(ox, 20);
      ctx.lineTo(ox, H - 24);
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.font = "700 11.5px 'Source Sans 3', sans-serif";
      ctx.fillText("X-Axis: True Vocal-Tract / Speech Feature (Real < 0 | Fake > 0)", 68, H - 8);
      ctx.save();
      ctx.translate(16, 215);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Y-Axis: MP3 / Phone Artifact Level", 0, 0);
      ctx.restore();

      // Plot Training Real (Blue circles, bottom-left)
      trainReal.forEach(([x, y]) => {
        ctx.fillStyle = "#1f77b4";
        ctx.beginPath();
        ctx.arc(ox + x * sx, oy - y * sy, 5.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Plot Training Fake (Red X, top-right)
      ctx.strokeStyle = "#d93a4b";
      ctx.lineWidth = 2.2;
      trainFake.forEach(([x, y]) => {
        const px = ox + x * sx;
        const py = oy - y * sy;
        ctx.beginPath();
        ctx.moveTo(px - 5, py - 5);
        ctx.lineTo(px + 5, py + 5);
        ctx.moveTo(px + 5, py - 5);
        ctx.lineTo(px - 5, py + 5);
        ctx.stroke();
      });

      if (scene === "1") {
        // Dashed oval showing spurious correlation in training data
        ctx.strokeStyle = "#e67e22";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(ox - 110, oy + 60);
        ctx.lineTo(ox + 110, oy - 60);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#e67e22";
        ctx.font = "700 12px 'Source Sans 3', sans-serif";
        ctx.fillText("Training Correlation: Fake Clips Also Have Higher Y!", ox - 125, 36);
      }

      if (scene === "2" || scene === "3") {
        // Show Unseen Test Real Audio on MP3 / Phone at Top-Left (Orange Squares)
        testRealMP3.forEach(([x, y], i) => {
          const px = ox + x * sx;
          const py = oy - y * sy;
          if (scene === "3") {
            // Draw vertical paired arrow from Clean WAV (bottom-left) to MP3 (top-left)
            const [rx, ry] = trainReal[i];
            ctx.strokeStyle = "rgba(17, 134, 89, 0.55)";
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(ox + rx * sx, oy - ry * sy);
            ctx.lineTo(px, py);
            ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.fillStyle = "#e67e22";
          ctx.fillRect(px - 5, py - 5, 10, 10);
        });
      }

      if (scene === "2") {
        // Draw Tilted SGD Boundary
        ctx.strokeStyle = "#d93a4b";
        ctx.lineWidth = 2.8;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(ox - 120, oy - 105);
        ctx.lineTo(ox + 120, oy + 105);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#d93a4b";
        ctx.font = "700 12px 'Source Sans 3', sans-serif";
        ctx.fillText("25-Epoch SGD Boundary (Tilted!)", ox - 185, 36);
        ctx.fillText("Real MP3 Test Clips Fall on FAKE Side!", ox - 195, 54);
      } else if (scene === "3") {
        // Draw Vertical 1-Pass OFP Boundary
        ctx.strokeStyle = "#118659";
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        ctx.moveTo(ox, 22);
        ctx.lineTo(ox, H - 26);
        ctx.stroke();
        ctx.fillStyle = "#118659";
        ctx.font = "700 12px 'Source Sans 3', sans-serif";
        ctx.fillText("1-Pass OFP Boundary (Locked Vertical: w_aug = 0)", ox - 195, 36);
      }

      // Right Panel: Visual Explanation & Bar Comparison
      const rx0 = 468;
      ctx.strokeStyle = "#e2ebe6";
      ctx.beginPath();
      ctx.moveTo(456, 16);
      ctx.lineTo(456, H - 16);
      ctx.stroke();

      const stepsInfo =
        scene === "1"
          ? {
              head: "Step 1: Why the Training Set Has a Shortcut",
              lines: [
                "• Blue Circles (Bottom-Left): Real training voices (mostly clean studio .WAV).",
                "• Red Crosses (Top-Right): Fake training voices (often carry codec/vocoder artifacts on the Y-axis).",
                "• Because Fake clips are shifted both right (X) and up (Y), any standard optimizer thinks Y (compression artifact) is a valid clue for 'Fake'!",
              ],
            }
          : scene === "2"
          ? {
              head: "Step 2: 25 Epochs of SGD / AdamW Learns the Tilted Boundary",
              lines: [
                "• Over 25 epochs, SGD/AdamW puts weight w_aug = +0.64 on the Y-axis, tilting the red dashed boundary.",
                "• Now look at the Orange Squares (Top-Left): these are REAL human voices uploaded as .MP3 or recorded over a phone call!",
                "• Because they sit high on the Y-axis, the tilted SGD boundary misclassifies them as Fake (6.95% Unseen EER)!",
              ],
            }
          : {
              head: "Step 3: 1-Pass Optimizer Locks w_aug = 0 in Closed Form",
              lines: [
                "• In 1 forward pass, we compress the blue studio clips into orange MP3 clips (green vertical dashed arrows).",
                "• That vertical arrow tells us the exact compression direction: c_aug = [0, +1].",
                "• Solving for weights with the hard lock c_aug^T w = 0 forces the boundary to be strictly vertical (4.12% Unseen EER in 1 pass)!",
              ],
            };

      ctx.fillStyle = scene === "2" ? "#d93a4b" : "#118659";
      ctx.font = "700 13.5px 'Source Sans 3', sans-serif";
      ctx.fillText(stepsInfo.head, rx0, 34);

      ctx.fillStyle = "#222";
      ctx.font = "12.5px 'Source Sans 3', sans-serif";
      let curY = 60;
      stepsInfo.lines.forEach((ln) => {
        const words = ln.split(" ");
        let line = "";
        words.forEach((w) => {
          if ((line + w).length > 58) {
            ctx.fillText(line, rx0, curY);
            line = "   " + w + " ";
            curY += 18;
          } else {
            line += w + " ";
          }
        });
        if (line) {
          ctx.fillText(line, rx0, curY);
          curY += 23;
        }
      });

      // Bottom Right Mini Bar Comparison
      const bars = [
        { label: "25-Epoch AdamW (Tilted Shortcut)", val: 6.95, col: "#d93a4b" },
        { label: "1-Pass OFP (Vertical Locked w_aug=0)", val: 4.12, col: "#118659" },
      ];
      bars.forEach((b, idx) => {
        const by = H - 58 + idx * 24;
        ctx.fillStyle = "#333";
        ctx.font = "600 11.5px 'Source Sans 3', sans-serif";
        ctx.fillText(b.label, rx0, by + 12);
        const bw = (b.val / 8.0) * 140;
        ctx.fillStyle = b.col;
        ctx.fillRect(rx0 + 215, by, bw, 15);
        ctx.fillStyle = "#111";
        ctx.font = "700 11.5px 'JetBrains Mono', monospace";
        ctx.fillText(`${b.val}% EER`, rx0 + 220 + bw, by + 12);
      });

      if (caption) {
        caption.textContent =
          scene === "1"
            ? "Blue Circles = Clean Real Voice | Red X = Fake Voice. Click Step 2 to see how 25-epoch SGD tilts the boundary."
            : scene === "2"
            ? "Orange Squares = Real Human Voices on MP3/Phone. The tilted SGD boundary puts them in the red Fake zone! Click Step 3 for 1-Pass OFP."
            : "1-Pass OFP uses the vertical paired arrows (Clean → MP3) to lock the boundary strictly vertical, keeping all Orange Squares on the Real side.";
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

      ctx.fillStyle = "rgba(217, 58, 75, 0.16)";
      ctx.beginPath();
      teachPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      for (let i = compPts.length - 1; i >= 0; i--) ctx.lineTo(compPts[i][0], compPts[i][1]);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = "#222";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      teachPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = alpha > 0.5 ? "#118659" : "#d93a4b";
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      compPts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.font = "600 12px 'Source Sans 3', sans-serif";
      ctx.fillText('Stop Consonant "p" Burst', x0 + plotW * 0.15, 26);
      ctx.fillText('Stop Consonant "t" Burst', x0 + plotW * 0.45, 26);
      ctx.fillText('Stop Consonant "k" Burst', x0 + plotW * 0.73, 26);

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
    initBlock2AWidget();
    initBlock2BWidget();
    initBlock3Circuit();
    initBlock4Scene();
    initBlock5Widget();
  });
})();
