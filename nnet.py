import numpy as np
import matplotlib.pyplot as plt


# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Mean Squared Error Loss and its derivative
def mean_squared_error_loss(y_true, y_pred):
    mse_loss = np.mean((y_true - y_pred) ** 2)
    return mse_loss

def mse_loss_derivative(y_true, y_pred):
    n = len(y_true)
    derivative = (-2/n) * np.sum(y_true - y_pred)
    return derivative    

# Binary Cross-Entropy Loss and its derivative
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predictions to avoid log(0) error
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid division by zero
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Layer classes as provided
class Layer:
    def forward(self, input_data):
        pass

    def backward(self, output_error, learning_rate):
        pass

class FCLayer(Layer):
    def __init__(self, input_size=None, output_size=None, weights=None, bias=None):
        if weights is not None and bias is not None:
            self.weights = weights
            self.bias = bias.reshape(1, -1)  # Ensure bias is 2D: (1, output_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / output_size)
            self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        # Ensure input_data is 2D
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # Ensure output_error is 2D
        if output_error.ndim == 1:
            output_error = output_error[np.newaxis, :]
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error

# Network class adjusted for binary classification
class Network:
    def __init__(self, loss_function, loss_derivative):
        self.layers = []
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
        self.loss_curve = []
        self.val_loss_curve = []  # For storing validation loss if any

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = input_data
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def train(self, x_train, y_train, epochs, learning_rate, x_val=None, y_val=None):
        for i in range(epochs):
            epoch_loss = 0.0
            for x, y in zip(x_train, y_train):
                output = self.predict(x[np.newaxis, :])
                error = self.loss_derivative(y, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                epoch_loss += self.loss_function(y, output)
            epoch_loss /= len(x_train)
            self.loss_curve.append(epoch_loss)

            # If validation data is provided, compute validation loss
            if x_val is not None and y_val is not None:
                val_loss = 0.0
                for x, y in zip(x_val, y_val):
                    val_output = self.predict(x[np.newaxis, :])
                    val_loss += self.loss_function(y, val_output)
                val_loss /= len(x_val)
                self.val_loss_curve.append(val_loss)

                print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss}, Validation Loss: {val_loss}")
            else:
                print(f"Epoch {i+1}/{epochs}, Loss: {epoch_loss}")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_curve, label='Training Loss')
        if self.val_loss_curve:  # Check if there are any validation losses to plot
            plt.plot(self.val_loss_curve, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()