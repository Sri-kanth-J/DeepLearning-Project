/**
 * Skin Disease Classifier — client-side logic
 * Handles: drag-and-drop, file preview, POST /predict, result rendering, Chart.js chart.
 */

"use strict";

// ------------------------------------------------------------------ DOM refs
const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("fileInput");
const previewBox  = document.getElementById("previewBox");
const previewImg  = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");
const clearBtn    = document.getElementById("clearBtn");
const analyseBtn  = document.getElementById("analyseBtn");
const btnLabel    = document.getElementById("btnLabel");
const btnSpinner  = document.getElementById("btnSpinner");
const errorBanner = document.getElementById("errorBanner");
const emptyState  = document.getElementById("emptyState");
const resultsDiv  = document.getElementById("results");

// Result elements
const predCard       = document.getElementById("predCard");
const predName       = document.getElementById("predName");
const predConfidence = document.getElementById("predConfidence");
const predBar        = document.getElementById("predBar");
const predUrgency    = document.getElementById("predUrgency");
const top3Grid       = document.getElementById("top3Grid");
const descBox        = document.getElementById("descBox");
const descTitle      = document.getElementById("descTitle");
const descText       = document.getElementById("descText");

let currentFile = null;
let chartInstance = null;

// ------------------------------------------------------------------ Drag and drop
dropzone.addEventListener("click",    () => fileInput.click());
dropzone.addEventListener("keydown",  (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave",()  => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop",     (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ------------------------------------------------------------------ Clear
clearBtn.addEventListener("click", () => {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewBox.hidden = true;
  dropzone.hidden = false;
  analyseBtn.disabled = true;
  hideResults();
  hideError();
});

// ------------------------------------------------------------------ Handle file
function handleFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  if (!validTypes.includes(file.type)) {
    showError("Unsupported file type. Please upload a JPG, PNG, BMP, or WebP image.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("File exceeds 16 MB limit.");
    return;
  }

  hideError();
  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
    dropzone.hidden = true;
    previewBox.hidden = false;
    analyseBtn.disabled = false;
    hideResults();
  };
  reader.readAsDataURL(file);
}

// ------------------------------------------------------------------ Analyse
analyseBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  setLoading(true);
  hideError();
  hideResults();

  const fd = new FormData();
  fd.append("image", currentFile);

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || `Server error (${resp.status})`);
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Network error — is the Flask server running?");
  } finally {
    setLoading(false);
  }
});

// ------------------------------------------------------------------ Render results
function renderResults(data) {
  const pred = data.prediction;

  // Primary card colour by urgency level
  predCard.className = "pred-card";
  const u = (pred.urgency || "").toLowerCase();
  if (u === "none" || u === "low")        predCard.classList.add("card-high");
  else if (u === "medium")                predCard.classList.add("card-medium");
  else if (u === "high")                  predCard.classList.add("card-low");

  predName.textContent       = pred.display_name;
  predConfidence.textContent = `Confidence: ${pred.confidence_pct}`;
  predBar.style.width        = `${(pred.confidence * 100).toFixed(1)}%`;
  predUrgency.textContent    = `Urgency: ${pred.urgency}`;

  // Top 3
  top3Grid.innerHTML = "";
  data.top3.forEach((item, i) => {
    const card = document.createElement("div");
    card.className = "top3-card";
    card.innerHTML = `
      <div class="top3-rank">#${item.rank}</div>
      <div class="top3-name">${item.display_name}</div>
      <div class="top3-confidence">${item.confidence_pct}</div>
    `;
    top3Grid.appendChild(card);
  });

  // Disease description
  if (pred.description) {
    descTitle.textContent = `About: ${pred.display_name}`;
    descText.textContent  = pred.description;
    descBox.hidden = false;
  } else {
    descBox.hidden = true;
  }

  // Bar chart (all classes)
  buildChart(data.all_classes);

  emptyState.hidden = true;
  resultsDiv.hidden = false;
}

// ------------------------------------------------------------------ Chart.js
function buildChart(allClasses) {
  const labels = allClasses.map(c => c.display_name);
  const values = allClasses.map(c => +(c.confidence * 100).toFixed(2));
  const colors = allClasses.map(c => {
    const v = c.confidence;
    if (v >= 0.6) return "rgba(37,99,235,0.85)";
    if (v >= 0.3) return "rgba(217,119,6,0.75)";
    return "rgba(100,116,139,0.55)";
  });

  const ctx = document.getElementById("predChart").getContext("2d");

  if (chartInstance) chartInstance.destroy();

  // Chart height scales with number of classes
  const chartHeight = Math.max(260, allClasses.length * 32);
  document.getElementById("predChart").parentElement.style.minHeight = `${chartHeight}px`;

  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Confidence (%)",
        data: values,
        backgroundColor: colors,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.x.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: { callback: v => `${v}%`, font: { size: 11 } },
          grid: { color: "rgba(0,0,0,0.06)" }
        },
        y: {
          ticks: { font: { size: 11 } },
          grid: { display: false }
        }
      }
    }
  });
}

// ------------------------------------------------------------------ Helpers
function setLoading(on) {
  analyseBtn.disabled = on;
  btnLabel.hidden     = on;
  btnSpinner.hidden   = !on;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function hideResults() {
  resultsDiv.hidden = true;
  emptyState.hidden = false;
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}
