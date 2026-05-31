# DeepShape: Vanilla JS CNN Model

This project is a browser-based Convolutional Neural Network (CNN) built entirely in **Vanilla JavaScript**, without the use of any external machine learning libraries (like TensorFlow.js). 

## Project Overview (Exercise 2)

The application demonstrates real-time client-side inference and training. It is divided into two main sections:
1. **Model Page (`index.html`)**: An interactive drawing canvas where the user can sketch a Circle, Square, or Triangle. The application processes the drawing through a custom-built CNN architecture to predict the shape. The user can also control the hyper-parameters and fine-tune the model in the browser.
2. **AI Glossary (`dictionary.html`)**: A comprehensive dictionary containing over 40 terms related to AI, Neural Networks, Convolution, and Training processes.

## Features
* **No Libraries**: The entire mathematical logic (Forward Pass, Backpropagation, MaxPooling, Convolution, Softmax, Cross-Entropy) is implemented from scratch.
* **Persistent Weights**: Pre-trained model weights are loaded and updated in the browser's `localStorage`, allowing the model to improve over time and persist between sessions.
* **Modern UI**: Designed with custom CSS (Glassmorphism, Dark Theme) for an intuitive and impressive user experience.
* **Interactive Canvas**: Users can draw in real-time, and the canvas data is automatically normalized and downsampled to a 28x28 grayscale matrix for the network input.

## How to Run
You can view and test the live model directly in your browser here:
**[Live Demo: CNN Shape Identifier](https://omershohat.github.io/CNN-Shape-Identifier/)**

Alternatively, you can simply open `index.html` locally in any modern web browser. No local server or build tools (like npm/webpack) are required.

## Directory Structure
* `index.html` - The CNN model sandbox and drawing canvas.
* `dictionary.html` - The AI/CNN glossary.
* `css/app.css` - The custom styling for the application.
* `js/cnn.js` - The core mathematical engine for the CNN.
* `js/app.js` - The logic connecting the UI, Canvas, and CNN engine.
* `js/dictionary.js` - The logic and data for the glossary.
* `js/pretrained-model.js` - The pre-trained JSON weights for the network.
