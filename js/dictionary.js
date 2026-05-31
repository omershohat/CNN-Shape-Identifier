const terms = [
  {
    term: "Artificial Intelligence (AI)",
    explanation: "A field of computer science that builds systems able to perform tasks that usually require human intelligence, such as classification, prediction, and pattern recognition."
  },
  {
    term: "Machine Learning",
    explanation: "A method where the computer improves its behavior from data instead of being programmed with every rule manually."
  },
  {
    term: "Deep Learning",
    explanation: "A machine-learning approach that uses neural networks with multiple layers to learn complex patterns from data."
  },
  {
    term: "Neural Network",
    explanation: "A model built from connected mathematical units called neurons. Each neuron receives inputs, applies weights and bias, and sends an output forward."
  },
  {
    term: "CNN / Convolutional Neural Network",
    explanation: "A neural network designed mainly for images. It uses convolution filters to detect local visual features such as edges, corners, curves, and shapes."
  },
  {
    term: "Input Layer",
    explanation: "The first part of the model. In this project, the input layer is a 28×28 grayscale image created from the drawing canvas."
  },
  {
    term: "Convolution",
    explanation: "An operation where a small filter slides across an image and calculates values that show where a specific feature appears."
  },
  {
    term: "Filter / Kernel",
    explanation: "A small matrix of trainable numbers used by a convolution layer. The filter learns to respond strongly to a visual pattern."
  },
  {
    term: "Filter Size",
    explanation: "The width and height of the filter, such as 3×3, 5×5, or 7×7. Larger filters see a larger part of the image at once."
  },
  {
    term: "Feature Map",
    explanation: "The output image created by applying a filter. A feature map highlights where a learned feature appears in the input."
  },
  {
    term: "Channel",
    explanation: "One layer of image data. A grayscale image has one channel; an RGB color image has three channels: red, green, and blue."
  },
  {
    term: "Padding",
    explanation: "Adding empty pixels around the image before convolution. This project uses same-style padding so feature maps stay 28×28."
  },
  {
    term: "Stride",
    explanation: "The number of pixels the filter moves each step. This project uses stride 1, meaning the filter moves one pixel at a time."
  },
  {
    term: "Pooling",
    explanation: "A downsampling operation that reduces the size of feature maps. This project explains pooling, although the implemented model keeps the maps full-size and flattens them."
  },
  {
    term: "Max Pooling",
    explanation: "A pooling method that takes the largest value from a small region. It keeps strong features while reducing image size."
  },
  {
    term: "Flatten",
    explanation: "The process of converting a 3D feature-map volume into one long vector so it can enter a dense layer."
  },
  {
    term: "Dense Layer / Fully Connected Layer",
    explanation: "A layer where every input value is connected to every neuron in the layer. In this project, dense layers perform final classification."
  },
  {
    term: "Neuron",
    explanation: "A mathematical unit that receives inputs, multiplies them by weights, adds a bias, and applies an activation function."
  },
  {
    term: "Weight",
    explanation: "A trainable number that controls how strongly one value influences another value in the model."
  },
  {
    term: "Bias",
    explanation: "A trainable number added to a neuron or filter result. It lets the model shift activation values up or down."
  },
  {
    term: "Activation Function",
    explanation: "A function applied to neuron outputs. It helps the network learn non-linear patterns rather than only straight-line relationships."
  },
  {
    term: "ReLU",
    explanation: "A common activation function that returns the value if it is positive and returns 0 if it is negative. It is used after convolution and dense calculations in this project."
  },
  {
    term: "Softmax",
    explanation: "A function that converts output scores into probabilities that sum to 1. This project uses softmax for Circle, Square, and Triangle probabilities."
  },
  {
    term: "Class",
    explanation: "One possible category the model can predict. This project has three classes: Circle, Square, and Triangle."
  },
  {
    term: "Classification",
    explanation: "The task of assigning an input to one of several categories. Here, the model classifies a drawing as one of three shapes."
  },
  {
    term: "Prediction / Inference",
    explanation: "Running a trained model on a new input to get an answer. Pressing Predict Drawing performs inference."
  },
  {
    term: "Confidence",
    explanation: "The probability assigned to the selected class. For example, 92% confidence means the model strongly prefers that class."
  },
  {
    term: "Forward Pass",
    explanation: "The calculation path from input image to final probabilities. It includes convolution, ReLU, flattening, dense layers, and softmax."
  },
  {
    term: "Loss",
    explanation: "A number that measures how wrong the model was. Lower loss usually means better predictions."
  },
  {
    term: "Cross-Entropy Loss",
    explanation: "A common loss function for classification. It gives a high penalty when the correct class receives low probability."
  },
  {
    term: "Training",
    explanation: "The process of repeatedly showing examples to the model and updating its weights to reduce loss."
  },
  {
    term: "Epoch",
    explanation: "One training cycle. In this project, each epoch processes a small batch of generated and user-drawn shape samples."
  },
  {
    term: "Batch",
    explanation: "A group of samples used during training. This project trains on a small number of samples per epoch to stay fast in the browser."
  },
  {
    term: "Learning Rate",
    explanation: "A value that controls how large each weight update is. Too high can make training unstable; too low can make learning slow."
  },
  {
    term: "Backpropagation",
    explanation: "The algorithm that calculates how each weight contributed to the error, starting from the output layer and moving backward."
  },
  {
    term: "Gradient",
    explanation: "The direction and size of change needed for a weight to reduce the loss. Training updates weights using gradients."
  },
  {
    term: "Gradient Clipping",
    explanation: "A safety technique that limits very large gradients. This project clips gradients to keep browser training more stable."
  },
  {
    term: "Accuracy",
    explanation: "The percentage of predictions that are correct. Synthetic Test reports accuracy on generated circle, square, and triangle images."
  },
  {
    term: "Overfitting",
    explanation: "A problem where the model memorizes training examples but performs poorly on new examples. Using varied generated shapes helps reduce it."
  },
  {
    term: "Underfitting",
    explanation: "A problem where the model is too simple or insufficiently trained, so it cannot learn the pattern well."
  },
  {
    term: "Dataset",
    explanation: "The collection of examples used for training or testing. This project creates a synthetic dataset in JavaScript and can also use drawn samples."
  },
  {
    term: "Training Sample",
    explanation: "One image with its correct label. A drawn circle labeled Circle is one training sample."
  },
  {
    term: "Label",
    explanation: "The correct answer attached to a training sample. Labels in this project are Circle, Square, and Triangle."
  },
  {
    term: "Synthetic Data",
    explanation: "Generated examples created by code instead of collected manually. The project generates random circles, squares, and triangles for training."
  },
  {
    term: "Normalization",
    explanation: "Converting values into a useful range. Canvas pixels are converted into grayscale values between 0 and 1."
  },
  {
    term: "Grayscale Image",
    explanation: "An image represented by one brightness value per pixel. The drawing canvas is converted into one grayscale channel."
  },
  {
    term: "Canvas",
    explanation: "An HTML element that allows drawing in the browser. This project uses canvas as the image editor for the model."
  },
  {
    term: "LocalStorage",
    explanation: "Browser storage that keeps data after refresh. This project saves model weights in LocalStorage so the checker receives a trained model."
  },
  {
    term: "Model Weights Persistence",
    explanation: "Saving the model's learned numbers so training is not lost. The project saves weights after every epoch."
  },
  {
    term: "Hyperparameter",
    explanation: "A setting chosen before or during training, such as learning rate, number of layers, filter size, filters, and epochs."
  },
  {
    term: "Architecture",
    explanation: "The structure of the neural network, including how many layers it has and how each layer is connected."
  },
  {
    term: "Output Layer",
    explanation: "The final layer of the network. This project's output layer has three values, one for each shape class."
  },
  {
    term: "Model Reset",
    explanation: "Returning the model to a known state. The reset button restores the supplied seed weights and saves them again."
  },
  {
    term: "No External Libraries",
    explanation: "The model is implemented with plain JavaScript only. There is no TensorFlow.js, no ML framework, and no imported CDN dependency."
  }
];

const dictionaryList = document.getElementById("dictionaryList");
const termCount = document.getElementById("termCount");
const searchInput = document.getElementById("searchInput");

function renderTerms(filterText = "") {
  const normalized = filterText.trim().toLowerCase();
  const filtered = terms.filter(item => {
    return item.term.toLowerCase().includes(normalized) ||
      item.explanation.toLowerCase().includes(normalized);
  });

  dictionaryList.innerHTML = "";
  filtered.forEach(item => {
    const card = document.createElement("article");
    card.className = "term-card";

    const title = document.createElement("h3");
    title.textContent = item.term;

    const explanation = document.createElement("p");
    explanation.textContent = item.explanation;

    card.append(title, explanation);
    dictionaryList.appendChild(card);
  });

  termCount.textContent = `${filtered.length} / ${terms.length} terms`;
}

searchInput.addEventListener("input", () => renderTerms(searchInput.value));
renderTerms();
