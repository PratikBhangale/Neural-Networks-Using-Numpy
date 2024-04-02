# Neural Network Implementation in Python

This Python script (`nnet.py`) provides a simple implementation of a neural network using NumPy for matrix operations and Matplotlib for plotting training and validation loss curves. It's designed for educational purposes to demonstrate the basics of neural networks, including layer creation, forward and backward propagation, and loss computation.

## Features

- **Activation Functions**: Includes implementation of `tanh` and `sigmoid` activation functions along with their derivatives.
- **Loss Functions**: Implements Mean Squared Error (MSE) and Binary Cross-Entropy as loss functions, providing both the loss computation and its derivative.
- **Layer Classes**: Defines `Layer` as a base class with specific implementations for fully connected layers (`FCLayer`) and activation layers (`ActivationLayer`).
- **Network Class**: Orchestrates the neural network operations, allowing the addition of layers, training on data, predictions, and loss visualization.

## Requirements

- NumPy: For numerical operations
- Matplotlib: For plotting loss curves

Make sure you have these libraries installed in your environment:

```sh
pip install numpy matplotlib
```

## Usage:

1. Define your network by creating an instance of the Network class and adding layers to it, including fully connected and activation layers.

2. Train your network on your dataset by calling the train method with your data, specifying the number of epochs and learning rate. Optionally, you can provide validation data to monitor overfitting.

3. Make predictions using the trained model by calling the predict method with new data.

4. Visualize the loss over training and, optionally, validation data by calling the plot_loss method.


## Example:

Here's a quick start example to create a simple neural network with one hidden layer:

```
import numpy as np
from nnet import Network, FCLayer, ActivationLayer, tanh, tanh_derivative, mean_squared_error_loss, mse_loss_derivative

# Initialize the network
net = Network(loss_function=mean_squared_error_loss, loss_derivative=mse_loss_derivative)

# Adding layers
net.add(FCLayer(input_size=2, output_size=3))  # Input layer
net.add(ActivationLayer(activation=tanh, activation_derivative=tanh_derivative))
net.add(FCLayer(input_size=3, output_size=1))  # Output layer
net.add(ActivationLayer(activation=tanh, activation_derivative=tanh_derivative))

# Training data
X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
Y_train = np.array([0, 1, 1, 0])

# Train the network
net.train(x_train=X_train, y_train=Y_train, epochs=1000, learning_rate=0.1)

# Predict on new data
print(net.predict(np.array([1, 1])))
```


