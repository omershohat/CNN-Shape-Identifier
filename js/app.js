let model = loadStoredModelOrSeed();
let userSamples = [];
let isTraining = false;

const canvas = document.getElementById("drawingCanvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const storageStatus = document.getElementById("storageStatus");
const clearCanvasBtn = document.getElementById("clearCanvasBtn");
const predictBtn = document.getElementById("predictBtn");
const trainBtn = document.getElementById("trainBtn");
const testBtn = document.getElementById("testBtn");
const freshModelBtn = document.getElementById("freshModelBtn");
const resetBtn = document.getElementById("resetBtn");
const addSampleBtn = document.getElementById("addSampleBtn");
const sampleLabel = document.getElementById("sampleLabel");
const sampleCounter = document.getElementById("sampleCounter");
const predictionResult = document.getElementById("predictionResult");
const probabilityBars = document.getElementById("probabilityBars");
const trainingLog = document.getElementById("trainingLog");
const modelInfo = document.getElementById("modelInfo");

const convLayersInput = document.getElementById("convLayersInput");
const filtersInput = document.getElementById("filtersInput");
const filterSizeInput = document.getElementById("filterSizeInput");
const hiddenInput = document.getElementById("hiddenInput");
const learningRateInput = document.getElementById("learningRateInput");
const epochsInput = document.getElementById("epochsInput");

function initializeCanvas() {
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);

  // Pen size:
  // 18 = thick pen
  // 12 = medium pen
  // 10 = good smaller pen
  // 8  = thin pen
  // 6  = very thin pen
  context.lineWidth = 10;

  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "black";
}

function getPointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

let isDrawing = false;
let lastPoint = null;

canvas.addEventListener("pointerdown", (event) => {
  isDrawing = true;
  lastPoint = getPointerPosition(event);
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!isDrawing || !lastPoint) return;

  const point = getPointerPosition(event);

  context.beginPath();
  context.moveTo(lastPoint.x, lastPoint.y);
  context.lineTo(point.x, point.y);
  context.stroke();

  lastPoint = point;
});

canvas.addEventListener("pointerup", (event) => {
  isDrawing = false;
  lastPoint = null;
  canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener("pointerleave", () => {
  isDrawing = false;
  lastPoint = null;
});

function readSettingsFromControls() {
  return normalizeSettings({
    convLayers: convLayersInput.value,
    filtersPerLayer: filtersInput.value,
    filterSize: filterSizeInput.value,
    hiddenNeurons: hiddenInput.value,
    learningRate: learningRateInput.value,
    epochs: epochsInput.value,
  });
}

function writeSettingsToControls(settings) {
  convLayersInput.value = settings.convLayers;
  filtersInput.value = settings.filtersPerLayer;
  filterSizeInput.value = settings.filterSize;
  hiddenInput.value = settings.hiddenNeurons;
  learningRateInput.value = settings.learningRate;
  epochsInput.value = settings.epochs || 6;
}

function updateModelInfo() {
  const settings = model.settings;
  const featureShape =
    model.finalFeatureShape || calculateFeatureShape(settings);

  const flatSize = featureShape.flatSize;

  const convWeightCount = model.convLayers.reduce((sum, layer) => {
    return (
      sum +
      layer.kernels.length *
        layer.kernels[0].length *
        layer.kernels[0][0].length *
        layer.kernels[0][0][0].length +
      layer.biases.length
    );
  }, 0);

  const dense1Count =
    model.dense1.weights.length * model.dense1.weights[0].length +
    model.dense1.biases.length;

  const dense2Count =
    model.dense2.weights.length * model.dense2.weights[0].length +
    model.dense2.biases.length;

  const total = convWeightCount + dense1Count + dense2Count;

  modelInfo.innerHTML = `
    <strong>Current architecture:</strong><br>
    Input: ${INPUT_SIZE}×${INPUT_SIZE} grayscale image<br>
    Convolution layers: ${settings.convLayers}<br>
    Filters per layer: ${settings.filtersPerLayer}<br>
    Filter size: ${settings.filterSize}×${settings.filterSize}<br>
    Feature map after pooling: ${featureShape.height}×${featureShape.width}×${featureShape.channels}<br>
    Flatten size after Conv/ReLU/MaxPool blocks: ${flatSize}<br>
    Dense hidden neurons: ${settings.hiddenNeurons}<br>
    Output classes: ${CLASS_NAMES.join(", ")}<br>
    Approximate stored trainable values: ${total.toLocaleString()}<br>
    CNN pipeline: Conv → ReLU → 2×2 MaxPool → Flatten → Dense → Softmax<br>
  `;

  storageStatus.textContent = localStorage.getItem(MODEL_STORAGE_KEY)
    ? "LocalStorage: saved"
    : "LocalStorage: empty";
}

function appendLog(message) {
  const time = new Date().toLocaleTimeString();
  trainingLog.textContent += `\n[${time}] ${message}`;
  trainingLog.scrollTop = trainingLog.scrollHeight;
}

function setTrainingButtonsDisabled(disabled) {
  isTraining = disabled;

  trainBtn.disabled = disabled;
  testBtn.disabled = disabled;
  freshModelBtn.disabled = disabled;
  resetBtn.disabled = disabled;
  predictBtn.disabled = disabled;

  if (addSampleBtn) {
    addSampleBtn.disabled = disabled;
  }
}

function createBlankInputVolume() {
  return createVolume(INPUT_SIZE, INPUT_SIZE, 1, 0);
}

function findDrawingBounds(imageData, width, height) {
  const data = imageData.data;

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;

      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const a = data[index + 3];

      const brightness = (r + g + b) / 3;

      if (a > 0 && brightness < 245) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX === -1 || maxY === -1) {
    return null;
  }

  const padding = 16;

  return {
    minX: Math.max(0, minX - padding),
    minY: Math.max(0, minY - padding),
    maxX: Math.min(width - 1, maxX + padding),
    maxY: Math.min(height - 1, maxY + padding),
  };
}

function canvasToInputVolume() {
  const sourceWidth = canvas.width;
  const sourceHeight = canvas.height;

  const sourceImageData = context.getImageData(0, 0, sourceWidth, sourceHeight);

  const bounds = findDrawingBounds(sourceImageData, sourceWidth, sourceHeight);

  if (!bounds) {
    return createBlankInputVolume();
  }

  const cropWidth = bounds.maxX - bounds.minX + 1;
  const cropHeight = bounds.maxY - bounds.minY + 1;

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cropWidth;
  cropCanvas.height = cropHeight;

  const cropContext = cropCanvas.getContext("2d", {
    willReadFrequently: true,
  });

  cropContext.fillStyle = "white";
  cropContext.fillRect(0, 0, cropWidth, cropHeight);

  cropContext.drawImage(
    canvas,
    bounds.minX,
    bounds.minY,
    cropWidth,
    cropHeight,
    0,
    0,
    cropWidth,
    cropHeight,
  );

  const targetShapeSize = 24;
  const scale = targetShapeSize / Math.max(cropWidth, cropHeight);

  const resizedWidth = Math.max(1, Math.round(cropWidth * scale));
  const resizedHeight = Math.max(1, Math.round(cropHeight * scale));

  const normalizedCanvas = document.createElement("canvas");
  normalizedCanvas.width = INPUT_SIZE;
  normalizedCanvas.height = INPUT_SIZE;

  const normalizedContext = normalizedCanvas.getContext("2d", {
    willReadFrequently: true,
  });

  normalizedContext.imageSmoothingEnabled = true;
  normalizedContext.fillStyle = "white";
  normalizedContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const pasteX = Math.floor((INPUT_SIZE - resizedWidth) / 2);
  const pasteY = Math.floor((INPUT_SIZE - resizedHeight) / 2);

  normalizedContext.drawImage(
    cropCanvas,
    0,
    0,
    cropWidth,
    cropHeight,
    pasteX,
    pasteY,
    resizedWidth,
    resizedHeight,
  );

  const normalizedImageData = normalizedContext.getImageData(
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE,
  );

  const data = normalizedImageData.data;
  const volume = createVolume(INPUT_SIZE, INPUT_SIZE, 1, 0);

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const index = (y * INPUT_SIZE + x) * 4;

      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];

      const brightness = (r + g + b) / 3;

      volume[y][x][0] = clamp(1 - brightness / 255, 0, 1);
    }
  }

  return volume;
}

function isCanvasBlank(input) {
  let sum = 0;

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      sum += input[y][x][0];
    }
  }

  return sum < 2;
}

function renderProbabilities(probabilities) {
  probabilityBars.innerHTML = "";

  probabilities.forEach((probability, index) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("strong");
    label.textContent = CLASS_NAMES[index];

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.round(probability * 100)}%`;

    track.appendChild(fill);

    const value = document.createElement("span");
    value.textContent = `${(probability * 100).toFixed(1)}%`;

    row.append(label, track, value);
    probabilityBars.appendChild(row);
  });
}

function predictCurrentDrawing() {
  const input = canvasToInputVolume();

  if (isCanvasBlank(input)) {
    predictionResult.textContent =
      "Prediction: the canvas is empty. Draw a circle, square, or triangle first.";
    probabilityBars.innerHTML = "";
    return;
  }

  const result = forwardModel(model, input, false);
  const className = CLASS_NAMES[result.predictedIndex];
  const confidence = result.probabilities[result.predictedIndex] * 100;

  predictionResult.textContent = `Prediction: ${className} (${confidence.toFixed(1)}% softmax score)`;

  renderProbabilities(result.probabilities);
}

function addCurrentDrawingAsSample() {
  if (!sampleLabel || !sampleCounter) {
    appendLog(
      "Cannot add sample: sample controls are not available on this page.",
    );
    return;
  }

  const input = canvasToInputVolume();

  if (isCanvasBlank(input)) {
    appendLog("Cannot add sample: canvas is empty.");
    return;
  }

  const labelIndex = parseInt(sampleLabel.value, 10);

  userSamples.push({
    input,
    labelIndex,
  });

  sampleCounter.textContent = `Drawn samples in memory: ${userSamples.length}`;

  appendLog(`Added drawn sample as ${CLASS_NAMES[labelIndex]}.`);
}

function pickTrainingSample(index) {
  if (userSamples.length > 0 && Math.random() < 0.25) {
    return userSamples[Math.floor(Math.random() * userSamples.length)];
  }

  const labelIndex = index % CLASS_NAMES.length;

  return {
    input: generateSyntheticSample(labelIndex),
    labelIndex,
  };
}

async function trainModel() {
  if (isTraining) return;

  const settings = readSettingsFromControls();

  if (!sameArchitecture(model, settings)) {
    model = createRandomModel(settings);
    appendLog(
      "Architecture changed, so a fresh random model was created before training.",
    );
  }

  model.settings.learningRate = settings.learningRate;
  model.settings.epochs = settings.epochs;

  setTrainingButtonsDisabled(true);

  trainingLog.textContent =
    "Browser fine-tuning started. The supplied trained weights are loaded; this optional local step continues training in JavaScript.";

  const samplesPerEpoch = 36;

  for (let epoch = 1; epoch <= settings.epochs; epoch++) {
    let totalLoss = 0;
    let correct = 0;

    for (let i = 0; i < samplesPerEpoch; i++) {
      const sample = pickTrainingSample(i + epoch);

      const result = trainOneSample(
        model,
        sample.input,
        sample.labelIndex,
        settings.learningRate,
      );

      totalLoss += result.loss;
      correct += result.correct;
    }

    const avgLoss = totalLoss / samplesPerEpoch;
    const accuracy = correct / samplesPerEpoch;

    appendLog(
      `Epoch ${epoch}/${settings.epochs} | loss=${avgLoss.toFixed(4)} | training accuracy=${(accuracy * 100).toFixed(1)}%`,
    );

    saveModel(model);
    updateModelInfo();

    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const test = testModelOnSyntheticData(model, 60);

  appendLog(
    `Saved fine-tuned weights to LocalStorage. Synthetic test accuracy=${(test.accuracy * 100).toFixed(1)}%, loss=${test.loss.toFixed(4)}.`,
  );

  setTrainingButtonsDisabled(false);
}

function testSyntheticData() {
  const result = testModelOnSyntheticData(model, 90);

  appendLog(
    `Synthetic test: accuracy=${(result.accuracy * 100).toFixed(1)}%, loss=${result.loss.toFixed(4)}.`,
  );
}

clearCanvasBtn.addEventListener("click", () => {
  initializeCanvas();
  predictionResult.textContent = "Prediction: draw a shape and click Predict.";
  probabilityBars.innerHTML = "";
});

predictBtn.addEventListener("click", predictCurrentDrawing);

if (addSampleBtn) {
  addSampleBtn.addEventListener("click", addCurrentDrawingAsSample);
}

trainBtn.addEventListener("click", trainModel);
testBtn.addEventListener("click", testSyntheticData);

freshModelBtn.addEventListener("click", () => {
  model = createRandomModel(readSettingsFromControls());
  saveModel(model);
  updateModelInfo();

  appendLog(
    "Fresh random model created and saved. Train it to improve predictions.",
  );
});

resetBtn.addEventListener("click", () => {
  model = resetToSeedModel();

  writeSettingsToControls(model.settings);
  updateModelInfo();

  appendLog(
    "Model reset to the supplied trained weights and saved back to LocalStorage.",
  );
});

initializeCanvas();
writeSettingsToControls(model.settings);
updateModelInfo();
renderProbabilities([0, 0, 0]);

appendLog(
  "Trained model loaded. Canvas preprocessing now crops, resizes, and centers drawings before prediction.",
);
