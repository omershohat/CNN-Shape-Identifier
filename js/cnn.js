const MODEL_STORAGE_KEY = "cnn_shape_model_trained_v1";
const INPUT_SIZE = 28;
const CLASS_NAMES = ["Circle", "Square", "Triangle"];
const POOL_SIZE = 2;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randomNormal(mean = 0, std = 1) {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function createVolume(height, width, channels, fill = 0) {
  const volume = new Array(height);
  for (let y = 0; y < height; y++) {
    volume[y] = new Array(width);
    for (let x = 0; x < width; x++) {
      volume[y][x] = new Array(channels).fill(fill);
    }
  }
  return volume;
}

function zeros1D(size) {
  return new Array(size).fill(0);
}

function zeros2D(rows, cols) {
  const result = new Array(rows);
  for (let r = 0; r < rows; r++) {
    result[r] = new Array(cols).fill(0);
  }
  return result;
}

function zeros4D(filters, channels, kernelSize) {
  const result = new Array(filters);
  for (let f = 0; f < filters; f++) {
    result[f] = new Array(channels);
    for (let c = 0; c < channels; c++) {
      result[f][c] = new Array(kernelSize);
      for (let ky = 0; ky < kernelSize; ky++) {
        result[f][c][ky] = new Array(kernelSize).fill(0);
      }
    }
  }
  return result;
}

function cloneModel(model) {
  return JSON.parse(JSON.stringify(model));
}

function normalizeSettings(settings) {
  return {
    convLayers: clamp(parseInt(settings.convLayers, 10) || 2, 1, 3),
    filtersPerLayer: clamp(parseInt(settings.filtersPerLayer, 10) || 12, 2, 16),
    filterSize: [3, 5, 7].includes(parseInt(settings.filterSize, 10))
      ? parseInt(settings.filterSize, 10)
      : 3,
    hiddenNeurons: clamp(parseInt(settings.hiddenNeurons, 10) || 64, 4, 128),
    learningRate: clamp(
      parseFloat(settings.learningRate) || 0.001,
      0.0001,
      0.1,
    ),
    epochs: clamp(parseInt(settings.epochs, 10) || 120, 1, 200),
  };
}

function calculateFeatureShape(settings) {
  const clean = normalizeSettings(settings);
  let height = INPUT_SIZE;
  let width = INPUT_SIZE;

  for (let i = 0; i < clean.convLayers; i++) {
    height = Math.floor(height / POOL_SIZE);
    width = Math.floor(width / POOL_SIZE);
  }

  return {
    height,
    width,
    channels: clean.filtersPerLayer,
    flatSize: height * width * clean.filtersPerLayer,
  };
}

function createRandomModel(settings) {
  const clean = normalizeSettings(settings);
  const featureShape = calculateFeatureShape(clean);
  const model = {
    inputSize: INPUT_SIZE,
    classes: CLASS_NAMES,
    architecture:
      "Conv -> ReLU -> 2x2 MaxPool blocks -> Flatten -> Dense ReLU -> Dense Softmax",
    settings: clean,
    finalFeatureShape: featureShape,
    convLayers: [],
    dense1: null,
    dense2: null,
    trainedAt: "fresh random model",
    trainingSummary:
      "Fresh random browser model. Use Train / Fine-Tune to update its weights.",
  };

  let inputChannels = 1;
  for (let layerIndex = 0; layerIndex < clean.convLayers; layerIndex++) {
    const fanIn = clean.filterSize * clean.filterSize * inputChannels;
    const scale = Math.sqrt(2 / fanIn);
    const kernels = zeros4D(
      clean.filtersPerLayer,
      inputChannels,
      clean.filterSize,
    );

    for (let f = 0; f < clean.filtersPerLayer; f++) {
      for (let c = 0; c < inputChannels; c++) {
        for (let ky = 0; ky < clean.filterSize; ky++) {
          for (let kx = 0; kx < clean.filterSize; kx++) {
            kernels[f][c][ky][kx] = randomNormal(0, scale);
          }
        }
      }
    }

    model.convLayers.push({
      type: "convolution",
      activation: "relu",
      pooling: "max_pool_2x2_stride_2",
      kernels,
      biases: zeros1D(clean.filtersPerLayer),
    });

    inputChannels = clean.filtersPerLayer;
  }

  const denseScale = Math.sqrt(2 / featureShape.flatSize);
  const dense1Weights = zeros2D(featureShape.flatSize, clean.hiddenNeurons);
  for (let i = 0; i < featureShape.flatSize; i++) {
    for (let h = 0; h < clean.hiddenNeurons; h++) {
      dense1Weights[i][h] = randomNormal(0, denseScale);
    }
  }

  const outputScale = Math.sqrt(2 / clean.hiddenNeurons);
  const dense2Weights = zeros2D(clean.hiddenNeurons, CLASS_NAMES.length);
  for (let h = 0; h < clean.hiddenNeurons; h++) {
    for (let o = 0; o < CLASS_NAMES.length; o++) {
      dense2Weights[h][o] = randomNormal(0, outputScale);
    }
  }

  model.dense1 = {
    type: "fully_connected",
    activation: "relu",
    weights: dense1Weights,
    biases: zeros1D(clean.hiddenNeurons),
  };

  model.dense2 = {
    type: "fully_connected",
    activation: "softmax",
    weights: dense2Weights,
    biases: zeros1D(CLASS_NAMES.length),
  };

  return model;
}

function getStorage() {
  if (typeof localStorage === "undefined") {
    return null;
  }
  return localStorage;
}

function saveModel(model) {
  const storage = getStorage();
  if (!storage) return;
  storage.setItem(MODEL_STORAGE_KEY, JSON.stringify(model));
}

function loadStoredModelOrSeed() {
  const storage = getStorage();
  if (storage) {
    const saved = storage.getItem(MODEL_STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed && parsed.architecture && parsed.finalFeatureShape) {
          return parsed;
        }
      } catch (error) {
        storage.removeItem(MODEL_STORAGE_KEY);
      }
    }
  }

  const seeded = cloneModel(PRETRAINED_MODEL);
  saveModel(seeded);
  return seeded;
}

function resetToSeedModel() {
  const seeded = cloneModel(PRETRAINED_MODEL);
  saveModel(seeded);
  return seeded;
}

function sameArchitecture(model, settings) {
  const clean = normalizeSettings(settings);

  if (!model || !model.settings || !Array.isArray(model.convLayers)) {
    return false;
  }

  return (
    model.settings.convLayers === clean.convLayers &&
    model.settings.filtersPerLayer === clean.filtersPerLayer &&
    model.settings.filterSize === clean.filterSize &&
    model.settings.hiddenNeurons === clean.hiddenNeurons &&
    model.convLayers.length === clean.convLayers &&
    model.dense1 &&
    model.dense2
  );
}

function relu(value) {
  return value > 0 ? value : 0;
}

function softmax(logits) {
  const maxValue = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxValue));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function convolutionForward(input, layer) {
  const height = input.length;
  const width = input[0].length;
  const inputChannels = input[0][0].length;
  const filters = layer.kernels.length;
  const kernelSize = layer.kernels[0][0].length;
  const padding = Math.floor(kernelSize / 2);
  const z = createVolume(height, width, filters, 0);
  const activation = createVolume(height, width, filters, 0);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let f = 0; f < filters; f++) {
        let sum = layer.biases[f];
        for (let c = 0; c < inputChannels; c++) {
          for (let ky = 0; ky < kernelSize; ky++) {
            for (let kx = 0; kx < kernelSize; kx++) {
              const inputY = y + ky - padding;
              const inputX = x + kx - padding;
              if (
                inputY >= 0 &&
                inputY < height &&
                inputX >= 0 &&
                inputX < width
              ) {
                sum += input[inputY][inputX][c] * layer.kernels[f][c][ky][kx];
              }
            }
          }
        }
        z[y][x][f] = sum;
        activation[y][x][f] = relu(sum);
      }
    }
  }

  return { z, activation };
}

function maxPool2x2Forward(input) {
  const inputHeight = input.length;
  const inputWidth = input[0].length;
  const channels = input[0][0].length;
  const outputHeight = Math.floor(inputHeight / POOL_SIZE);
  const outputWidth = Math.floor(inputWidth / POOL_SIZE);
  const output = createVolume(outputHeight, outputWidth, channels, 0);
  const switches = createVolume(outputHeight, outputWidth, channels, 0);

  for (let y = 0; y < outputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      for (let c = 0; c < channels; c++) {
        let maxValue = -Infinity;
        let maxIndex = 0;

        for (let py = 0; py < POOL_SIZE; py++) {
          for (let px = 0; px < POOL_SIZE; px++) {
            const inputY = y * POOL_SIZE + py;
            const inputX = x * POOL_SIZE + px;
            const value = input[inputY][inputX][c];
            if (value > maxValue) {
              maxValue = value;
              maxIndex = py * POOL_SIZE + px;
            }
          }
        }

        output[y][x][c] = maxValue;
        switches[y][x][c] = maxIndex;
      }
    }
  }

  return { output, switches, inputHeight, inputWidth };
}

function maxPool2x2Backward(
  dOutput,
  switches,
  inputHeight,
  inputWidth,
  channels,
) {
  const dInput = createVolume(inputHeight, inputWidth, channels, 0);

  for (let y = 0; y < dOutput.length; y++) {
    for (let x = 0; x < dOutput[0].length; x++) {
      for (let c = 0; c < channels; c++) {
        const maxIndex = switches[y][x][c];
        const py = Math.floor(maxIndex / POOL_SIZE);
        const px = maxIndex % POOL_SIZE;
        dInput[y * POOL_SIZE + py][x * POOL_SIZE + px][c] += dOutput[y][x][c];
      }
    }
  }

  return dInput;
}

function flattenVolume(volume) {
  const flat = [];
  for (let y = 0; y < volume.length; y++) {
    for (let x = 0; x < volume[0].length; x++) {
      for (let c = 0; c < volume[0][0].length; c++) {
        flat.push(volume[y][x][c]);
      }
    }
  }
  return flat;
}

function unflattenToVolume(flat, height, width, channels) {
  const volume = createVolume(height, width, channels, 0);
  let index = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < channels; c++) {
        volume[y][x][c] = flat[index++];
      }
    }
  }
  return volume;
}

function forwardModel(model, input, keepCache = false) {
  const layerCaches = [];
  let current = input;

  for (const layer of model.convLayers) {
    const conv = convolutionForward(current, layer);
    const pool = maxPool2x2Forward(conv.activation);

    if (keepCache) {
      layerCaches.push({
        input: current,
        convZ: conv.z,
        reluActivation: conv.activation,
        poolSwitches: pool.switches,
        poolInputHeight: pool.inputHeight,
        poolInputWidth: pool.inputWidth,
      });
    }

    current = pool.output;
  }

  const flat = flattenVolume(current);
  const hiddenZ = zeros1D(model.settings.hiddenNeurons);
  const hidden = zeros1D(model.settings.hiddenNeurons);

  for (let h = 0; h < model.settings.hiddenNeurons; h++) {
    let sum = model.dense1.biases[h];
    for (let i = 0; i < flat.length; i++) {
      sum += flat[i] * model.dense1.weights[i][h];
    }
    hiddenZ[h] = sum;
    hidden[h] = relu(sum);
  }

  const logits = zeros1D(CLASS_NAMES.length);
  for (let o = 0; o < CLASS_NAMES.length; o++) {
    let sum = model.dense2.biases[o];
    for (let h = 0; h < hidden.length; h++) {
      sum += hidden[h] * model.dense2.weights[h][o];
    }
    logits[o] = sum;
  }

  const probabilities = softmax(logits);
  const predictedIndex = probabilities.indexOf(Math.max(...probabilities));

  if (!keepCache) {
    return { probabilities, predictedIndex };
  }

  return {
    probabilities,
    predictedIndex,
    cache: {
      layerCaches,
      finalFeatureMap: current,
      flat,
      hiddenZ,
      hidden,
    },
  };
}

function trainOneSample(model, input, labelIndex, learningRate) {
  const result = forwardModel(model, input, true);
  const probabilities = result.probabilities;
  const cache = result.cache;
  const loss = -Math.log(Math.max(probabilities[labelIndex], 1e-12));
  const dLogits = probabilities.slice();
  dLogits[labelIndex] -= 1;

  const dDense2Biases = dLogits.slice();
  const dHidden = zeros1D(model.settings.hiddenNeurons);

  for (let h = 0; h < model.settings.hiddenNeurons; h++) {
    for (let o = 0; o < CLASS_NAMES.length; o++) {
      dHidden[h] += model.dense2.weights[h][o] * dLogits[o];
    }
  }

  for (let h = 0; h < model.settings.hiddenNeurons; h++) {
    for (let o = 0; o < CLASS_NAMES.length; o++) {
      const gradient = cache.hidden[h] * dLogits[o];
      model.dense2.weights[h][o] -= learningRate * clamp(gradient, -5, 5);
    }
  }

  for (let o = 0; o < CLASS_NAMES.length; o++) {
    model.dense2.biases[o] -= learningRate * clamp(dDense2Biases[o], -5, 5);
  }

  const dHiddenZ = zeros1D(model.settings.hiddenNeurons);
  for (let h = 0; h < model.settings.hiddenNeurons; h++) {
    dHiddenZ[h] = cache.hiddenZ[h] > 0 ? dHidden[h] : 0;
  }

  const dFlat = zeros1D(cache.flat.length);
  for (let i = 0; i < cache.flat.length; i++) {
    for (let h = 0; h < model.settings.hiddenNeurons; h++) {
      dFlat[i] += model.dense1.weights[i][h] * dHiddenZ[h];
    }
  }

  for (let i = 0; i < cache.flat.length; i++) {
    for (let h = 0; h < model.settings.hiddenNeurons; h++) {
      const gradient = cache.flat[i] * dHiddenZ[h];
      model.dense1.weights[i][h] -= learningRate * clamp(gradient, -5, 5);
    }
  }

  for (let h = 0; h < model.settings.hiddenNeurons; h++) {
    model.dense1.biases[h] -= learningRate * clamp(dHiddenZ[h], -5, 5);
  }

  const finalShape =
    model.finalFeatureShape || calculateFeatureShape(model.settings);
  let dCurrent = unflattenToVolume(
    dFlat,
    finalShape.height,
    finalShape.width,
    finalShape.channels,
  );

  for (
    let layerIndex = model.convLayers.length - 1;
    layerIndex >= 0;
    layerIndex--
  ) {
    const layer = model.convLayers[layerIndex];
    const layerCache = cache.layerCaches[layerIndex];
    const previousActivation = layerCache.input;
    const convZ = layerCache.convZ;
    const inputHeight = previousActivation.length;
    const inputWidth = previousActivation[0].length;
    const inputChannels = previousActivation[0][0].length;
    const filters = layer.kernels.length;
    const kernelSize = layer.kernels[0][0].length;
    const padding = Math.floor(kernelSize / 2);

    const dAfterPool = maxPool2x2Backward(
      dCurrent,
      layerCache.poolSwitches,
      layerCache.poolInputHeight,
      layerCache.poolInputWidth,
      filters,
    );

    const dConvZ = createVolume(
      layerCache.poolInputHeight,
      layerCache.poolInputWidth,
      filters,
      0,
    );
    for (let y = 0; y < dAfterPool.length; y++) {
      for (let x = 0; x < dAfterPool[0].length; x++) {
        for (let f = 0; f < filters; f++) {
          dConvZ[y][x][f] = convZ[y][x][f] > 0 ? dAfterPool[y][x][f] : 0;
        }
      }
    }

    const dKernels = zeros4D(filters, inputChannels, kernelSize);
    const dBiases = zeros1D(filters);
    const dPrevious = createVolume(inputHeight, inputWidth, inputChannels, 0);

    for (let y = 0; y < inputHeight; y++) {
      for (let x = 0; x < inputWidth; x++) {
        for (let f = 0; f < filters; f++) {
          const gradient = dConvZ[y][x][f];
          dBiases[f] += gradient;

          for (let c = 0; c < inputChannels; c++) {
            for (let ky = 0; ky < kernelSize; ky++) {
              for (let kx = 0; kx < kernelSize; kx++) {
                const inputY = y + ky - padding;
                const inputX = x + kx - padding;
                if (
                  inputY >= 0 &&
                  inputY < inputHeight &&
                  inputX >= 0 &&
                  inputX < inputWidth
                ) {
                  dKernels[f][c][ky][kx] +=
                    previousActivation[inputY][inputX][c] * gradient;
                  dPrevious[inputY][inputX][c] +=
                    layer.kernels[f][c][ky][kx] * gradient;
                }
              }
            }
          }
        }
      }
    }

    const normalizer = 1 / (inputHeight * inputWidth);
    for (let f = 0; f < filters; f++) {
      layer.biases[f] -= learningRate * clamp(dBiases[f] * normalizer, -5, 5);
      for (let c = 0; c < inputChannels; c++) {
        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            layer.kernels[f][c][ky][kx] -=
              learningRate * clamp(dKernels[f][c][ky][kx] * normalizer, -5, 5);
          }
        }
      }
    }

    dCurrent = dPrevious;
  }

  model.trainedAt = new Date().toISOString();
  model.trainingSummary =
    "Browser fine-tuned CNN with Conv, ReLU, MaxPool, Dense, Softmax and backpropagation.";

  return {
    loss,
    correct: result.predictedIndex === labelIndex ? 1 : 0,
  };
}

function imageToVolume(image2D) {
  const volume = createVolume(INPUT_SIZE, INPUT_SIZE, 1, 0);
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      volume[y][x][0] = image2D[y][x];
    }
  }
  return volume;
}

function distancePointToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const denominator = abx * abx + aby * aby;
  const t =
    denominator === 0 ? 0 : clamp((apx * abx + apy * aby) / denominator, 0, 1);
  const cx = ax + t * abx;
  const cy = ay + t * aby;
  return Math.hypot(px - cx, py - cy);
}

function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const hasNegative = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPositive = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNegative && hasPositive);
}

function rotatePoint(x, y, centerX, centerY, angle) {
  const dx = x - centerX;
  const dy = y - centerY;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: centerX + dx * cos - dy * sin,
    y: centerY + dx * sin + dy * cos,
  };
}

function generateSyntheticSample(labelIndex) {
  const image = new Array(INPUT_SIZE);
  for (let y = 0; y < INPUT_SIZE; y++) {
    image[y] = new Array(INPUT_SIZE).fill(0);
  }

  const filled = Math.random() < 0.35;

  if (labelIndex === 0) {
    const centerX = INPUT_SIZE / 2 + (Math.random() * 4 - 2);
    const centerY = INPUT_SIZE / 2 + (Math.random() * 4 - 2);
    const radius = 7 + Math.random() * 2.5;
    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        const distance = Math.hypot(x - centerX, y - centerY);
        if (
          (filled && distance <= radius) ||
          (!filled && Math.abs(distance - radius) <= 1.7)
        ) {
          image[y][x] = 1;
        }
      }
    }
  }

  if (labelIndex === 1) {
    const size = Math.floor(14 + Math.random() * 5);
    const startX = Math.floor(4 + Math.random() * (INPUT_SIZE - size - 7));
    const startY = Math.floor(4 + Math.random() * (INPUT_SIZE - size - 7));
    const angle = Math.random() * 0.3 - 0.15;
    const centerX = startX + size / 2;
    const centerY = startY + size / 2;

    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        const p = rotatePoint(x, y, centerX, centerY, -angle);
        const insideX = p.x >= startX && p.x <= startX + size;
        const insideY = p.y >= startY && p.y <= startY + size;
        const nearBorder =
          Math.min(
            Math.abs(p.x - startX),
            Math.abs(p.x - (startX + size)),
            Math.abs(p.y - startY),
            Math.abs(p.y - (startY + size)),
          ) <= 1.6;
        if (insideX && insideY && (filled || nearBorder)) {
          image[y][x] = 1;
        }
      }
    }
  }

  if (labelIndex === 2) {
    const ax = INPUT_SIZE / 2 + (Math.random() * 4 - 2);
    const ay = 4 + Math.random() * 3;
    const bx = 5 + Math.random() * 2;
    const by = 21 + Math.random() * 3;
    const cx = 21 + Math.random() * 2;
    const cy = 21 + Math.random() * 3;

    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        const px = x + 0.5;
        const py = y + 0.5;
        const inside = pointInTriangle(px, py, ax, ay, bx, by, cx, cy);
        const edgeDistance = Math.min(
          distancePointToSegment(px, py, ax, ay, bx, by),
          distancePointToSegment(px, py, ax, ay, cx, cy),
          distancePointToSegment(px, py, bx, by, cx, cy),
        );
        if ((filled && inside) || (!filled && edgeDistance <= 1.7)) {
          image[y][x] = 1;
        }
      }
    }
  }

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      image[y][x] = clamp(image[y][x] + Math.random() * 0.015, 0, 1);
    }
  }

  return imageToVolume(image);
}

function testModelOnSyntheticData(model, sampleCount = 60) {
  let correct = 0;
  let totalLoss = 0;

  for (let i = 0; i < sampleCount; i++) {
    const labelIndex = i % CLASS_NAMES.length;
    const input = generateSyntheticSample(labelIndex);
    const result = forwardModel(model, input, false);
    totalLoss += -Math.log(Math.max(result.probabilities[labelIndex], 1e-12));
    if (result.predictedIndex === labelIndex) correct++;
  }

  return {
    accuracy: correct / sampleCount,
    loss: totalLoss / sampleCount,
  };
}

if (typeof module !== "undefined") {
  module.exports = {
    MODEL_STORAGE_KEY,
    INPUT_SIZE,
    CLASS_NAMES,
    normalizeSettings,
    calculateFeatureShape,
    createRandomModel,
    forwardModel,
    trainOneSample,
    generateSyntheticSample,
    testModelOnSyntheticData,
    cloneModel,
  };
}
