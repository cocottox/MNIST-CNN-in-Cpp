# MNIST CNN in Cpp - by Cocottox 

This project is a simple handwritten-digit recognition experiment built around the MNIST dataset. It combines:

- a Python prototype that trains a small CNN using NumPy
- a C++ exploration for reading MNIST files and testing the dataset loading logic
- a basic implementation of CNN components such as convolution, ReLU, max-pooling, flattening, and fully connected layers

The goal is to learn how a convolutional neural network works from scratch and to recognize digits from 0 to 9 using the classic MNIST dataset.

## Project overview

The project is divided into several files:

- `Python_CNN.py` — main training script for the CNN
- `main.cpp` — C++ program for reading MNIST data and printing a sample image
- `ConvNeuralNetwork.cpp` — exploratory CNN implementation in C++
- `mnist_reader.h` / `mnist_reader.cpp` — utilities to parse MNIST IDX files
- `MNIST_ORG/` — folder containing the dataset files

## What the Python program does

The script in `Python_CNN.py` does the following:

1. Reads MNIST images from `train-images.idx3-ubyte`
2. Reads labels from `train-labels.idx1-ubyte`
3. Normalizes pixel values from `0..255` to `0..1`
4. Creates a tiny CNN with:
   - convolution layer
   - ReLU activation
   - max-pooling layer
   - flattening
   - fully connected output layer
5. Uses softmax to convert outputs into probabilities
6. Computes cross-entropy loss
7. Runs backpropagation to update weights
8. Saves trained weights to `cnn_MINST_model.npz`

## Model structure

The network is intentionally simple and educational. It contains:

- Input image: 28x28x1 grayscale
- Convolution layer: 4 filters of size 3x3
- ReLU activation
- Max pooling: 2x2
- Flattening to a vector of 676 values
- Fully connected layer to 10 output classes

This architecture is enough to learn simple patterns for digit classification and is a good beginner-level CNN implementation.

## Training flow

The training loop inside `Python_CNN.py` performs this sequence for each image:

1. Forward pass through convolution
2. ReLU
3. Max pooling
4. Flatten
5. Fully connected layer
6. Softmax over 10 output classes
7. Loss calculation with target digit label
8. Backpropagation through the network
9. Weight update using gradient descent

The code updates the model by computing gradients and subtracting a fraction of them from the current weights.

## Dataset format

MNIST files are stored in IDX format:

- `train-images.idx3-ubyte` contains the training images
- `train-labels.idx1-ubyte` contains the corresponding labels

Each image is a 28x28 grayscale matrix; labels range from 0 to 9.

## Requirements

For the Python version, install:

```bash
pip install numpy matplotlib scipy
```

Make sure the dataset directory exists and contains the MNIST files.

## How to run the Python version

From the project folder:

```bash
python Python_CNN.py
```

The script expects a folder called `MNIST_ORG` in the same directory containing:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`

If the dataset is correctly placed, the script will train the model and save:

```bash
cnn_MINST_model.npz
```

## Functionality summary

This project teaches and demonstrates:

- reading binary dataset files
- image preprocessing
- convolution and pooling
- neural network training from scratch
- backpropagation
- probability output for classification
- saving learned weights

## Notes and limitations

This is a learning project, not a production-ready deep learning pipeline. Some limitations include:

- training is intentionally minimal
- only 2 epochs are used in the script
- the implementation is written in pure NumPy
- the C++ code is a partial prototype and not a full CNN training pipeline yet
- the program depends on the dataset being in the expected folder path

## Possible improvements

Future enhancements could include:

- using a proper training/test split
- implementing mini-batch training
- adding dropout or regularization
- supporting more convolutional layers
- creating a full C++ CNN implementation
- evaluating accuracy on the validation/test set

## License

This project is intended for educational and experimental use.

## Author note

This repository is a practical example of how a CNN can be built step by step in Python using only NumPy, while also exploring a C++ implementation for MNIST dataset handling and neural network basics.
