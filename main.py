import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import json


class ANN_2_2_1:
    """
    A 2-2-1 Artificial Neural Network implementation using PyTorch

    Attributes:
        input_size (int): Number of input neurons (2)
        hidden_size (int): Number of hidden neurons (2)
        output_size (int): Number of output neurons (1)
        learning_rate (float): Learning rate for gradient descent
        input_normalizer (Normalizer): Normalizer for input data
        output_normalizer (Normalizer): Normalizer for output data
        model (torch.nn.Sequential): PyTorch model representing the network
        error_history (list): List to store error values during training
        error_function (str): The error function to use ('mse', 'mae', or 'bce')
    """

    def __init__(self, learning_rate=0.01):
        """
        Initialize the ANN with specified parameters

        Args:
            learning_rate (float): Learning rate for gradient descent
        """
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1
        self.learning_rate = learning_rate
        self.input_normalizer = Normalizer()
        self.output_normalizer = Normalizer()
        self.error_history = []
        self.error_function = 'mse'  # Default error function

        # Initialize the model with random weights
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the neural network model with random weights.
        Uses ReLU activation in the hidden layer and Sigmoid in the output layer.
        """
        # Create a sequential model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),  # ReLU activation for hidden layer
            torch.nn.Linear(self.hidden_size, self.output_size),
            torch.nn.Sigmoid()  # Sigmoid activation for output layer
        )

        # Initialize weights with small random values
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Better for ReLU
                torch.nn.init.zeros_(layer.bias)

    def set_error_function(self, error_function):
        """
        Set the error function to use for training

        Args:
            error_function (str): Error function to use ('mse', 'mae', or 'bce')
        """
        if error_function.lower() not in ['mse', 'mae', 'bce']:
            raise ValueError("Error function must be one of: 'mse', 'mae', 'bce'")

        self.error_function = error_function.lower()

    def calculate_error(self, outputs, targets):
        """
        Calculate error based on the selected error function

        Args:
            outputs (torch.Tensor): Network outputs
            targets (torch.Tensor): Target values

        Returns:
            torch.Tensor: Calculated error
        """
        if self.error_function == 'mse':
            return torch.mean((outputs - targets) ** 2)
        elif self.error_function == 'mae':
            return torch.mean(torch.abs(outputs - targets))
        elif self.error_function == 'bce':
            return torch.nn.functional.binary_cross_entropy(outputs, targets)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

    def train_incremental(self, inputs, targets, epochs=1000, error_threshold=0.001, max_retries=5):
        """
        Train the network using incremental (online) training

        Args:
            inputs (numpy.ndarray): Input data
            targets (numpy.ndarray): Target data
            epochs (int): Maximum number of training epochs
            error_threshold (float): Error threshold for convergence
            max_retries (int): Maximum number of retries if convergence fails

        Returns:
            bool: True if training converged, False otherwise
        """
        # Reset error history
        self.error_history = []

        # Start plotting
        self._setup_plot()

        for retry in range(max_retries):
            # Re-initialize weights if this is a retry
            if retry > 0:
                print(f"Training did not converge, retry {retry}/{max_retries}")
                self.initialize_model()
                self.error_history = []

            # Train for specified number of epochs
            for epoch in range(epochs):
                epoch_error = 0.0

                # Process each sample individually
                for i in range(len(inputs)):
                    # Convert to PyTorch tensors
                    x = torch.tensor(inputs[i:i + 1], dtype=torch.float32)
                    y = torch.tensor(targets[i:i + 1], dtype=torch.float32)

                    # Forward pass
                    y_pred = self.forward(x)

                    # Compute loss
                    loss = self.calculate_error(y_pred, y)
                    epoch_error += loss.item()

                    # Backward pass
                    self.model.zero_grad()
                    loss.backward()

                    # Manual weight update
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= self.learning_rate * param.grad

                # Calculate average error for the epoch
                avg_error = epoch_error / len(inputs)
                self.error_history.append(avg_error)

                # Update plot
                self._update_plot()

                # Check for convergence
                if avg_error < error_threshold:
                    print(f"Training converged in {epoch + 1} epochs with error {avg_error:.6f}")
                    return True

                # Print progress every 100 epochs
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Error: {avg_error:.6f}")

        print("Training failed to converge after maximum retries")
        return False

    def train_batch(self, inputs, targets, epochs=1000, error_threshold=0.001, max_retries=5):
        """
        Train the network using batch training

        Args:
            inputs (numpy.ndarray): Input data
            targets (numpy.ndarray): Target data
            epochs (int): Maximum number of training epochs
            error_threshold (float): Error threshold for convergence
            max_retries (int): Maximum number of retries if convergence fails

        Returns:
            bool: True if training converged, False otherwise
        """
        # Reset error history
        self.error_history = []

        # Start plotting
        self._setup_plot()

        # Convert data to PyTorch tensors
        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        for retry in range(max_retries):
            # Re-initialize weights if this is a retry
            if retry > 0:
                print(f"Training did not converge, retry {retry}/{max_retries}")
                self.initialize_model()
                self.error_history = []

            # Train for specified number of epochs
            for epoch in range(epochs):
                # Forward pass
                y_pred = self.forward(x)

                # Compute loss
                loss = self.calculate_error(y_pred, y)

                # Backward pass
                self.model.zero_grad()
                loss.backward()

                # Manual weight update
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= self.learning_rate * param.grad

                # Record error
                error = loss.item()
                self.error_history.append(error)

                # Update plot
                self._update_plot()

                # Check for convergence
                if error < error_threshold:
                    print(f"Training converged in {epoch + 1} epochs with error {error:.6f}")
                    return True

                # Print progress every 100 epochs
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Error: {error:.6f}")

        print("Training failed to converge after maximum retries")
        return False

    def _setup_plot(self):
        """Set up the plot for visualizing error convergence"""
        plt.figure(figsize=(10, 6))
        plt.ion()  # Turn on interactive mode
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Training Error Convergence ({self.error_function.upper()})')
        plt.grid(True)
        plt.show(block=False)

    def _update_plot(self):
        """Update the plot with current error values"""
        plt.clf()
        plt.plot(self.error_history, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Training Error Convergence ({self.error_function.upper()})')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    def test(self, inputs, targets):
        """
        Test the network on new data

        Args:
            inputs (numpy.ndarray): Input test data
            targets (numpy.ndarray): Target test data

        Returns:
            tuple: (outputs, error)
        """
        # Convert to PyTorch tensors
        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        # Forward pass
        with torch.no_grad():
            y_pred = self.forward(x)

        # Calculate error
        error = self.calculate_error(y_pred, y).item()

        # Convert output to numpy array
        outputs = y_pred.numpy()

        return outputs, error

    def save_model(self, filename='ann_model.pt'):
        """
        Save the neural network model to a file

        Args:
            filename (str): Filename to save the model
        """
        # Save model state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_normalizer': {
                'min_vals': self.input_normalizer.min_vals,
                'max_vals': self.input_normalizer.max_vals
            },
            'output_normalizer': {
                'min_vals': self.output_normalizer.min_vals,
                'max_vals': self.output_normalizer.max_vals
            },
            'error_function': self.error_function,
            'learning_rate': self.learning_rate
        }

        torch.save(model_state, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='ann_model.pt'):
        """
        Load the neural network model from a file

        Args:
            filename (str): Filename to load the model from
        """
        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")

        # Load model state
        model_state = torch.load(filename)

        # Restore model
        self.model.load_state_dict(model_state['model_state_dict'])

        # Restore normalizers
        self.input_normalizer.min_vals = model_state['input_normalizer']['min_vals']
        self.input_normalizer.max_vals = model_state['input_normalizer']['max_vals']
        self.output_normalizer.min_vals = model_state['output_normalizer']['min_vals']
        self.output_normalizer.max_vals = model_state['output_normalizer']['max_vals']

        # Restore other parameters
        self.error_function = model_state['error_function']
        self.learning_rate = model_state['learning_rate']

        print(f"Model loaded from {filename}")


class Normalizer:
    """
    Class for normalizing and denormalizing data

    Attributes:
        min_vals (numpy.ndarray): Minimum values for each feature
        max_vals (numpy.ndarray): Maximum values for each feature
    """

    def __init__(self):
        """Initialize the normalizer"""
        self.min_vals = None
        self.max_vals = None

    def fit(self, data):
        """
        Fit the normalizer to the data

        Args:
            data (numpy.ndarray): Data to fit the normalizer
        """
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)

    def normalize(self, data):
        """
        Normalize the data to [0, 1] range

        Args:
            data (numpy.ndarray): Data to normalize

        Returns:
            numpy.ndarray: Normalized data
        """
        if self.min_vals is None or self.max_vals is None:
            self.fit(data)

        # Handle case where min == max (constant feature)
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero

        return (data - self.min_vals) / range_vals

    def denormalize(self, normalized_data):
        """
        Denormalize the data back to original scale

        Args:
            normalized_data (numpy.ndarray): Normalized data

        Returns:
            numpy.ndarray: Denormalized data
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalizer is not fitted. Call fit() first.")

        return normalized_data * (self.max_vals - self.min_vals) + self.min_vals


def generate_heat_transfer_data(n_samples=1000):
    """
    Generate data for a simplified heat transfer problem

    Formula: final_temp = initial_temp + 0.5 * initial_temp * sin(0.1 * duration) + 0.3 * duration

    Args:
        n_samples (int): Number of samples to generate

    Returns:
        tuple: (inputs, outputs)
    """
    # Generate random initial temperatures between 20°C and 100°C
    initial_temp = np.random.uniform(20, 100, n_samples)

    # Generate random durations between 0 and 60 seconds
    duration = np.random.uniform(0, 60, n_samples)

    # Calculate final temperature
    final_temp = initial_temp + 0.5 * initial_temp * np.sin(0.1 * duration) + 0.3 * duration

    # Stack inputs and reshape outputs
    inputs = np.column_stack((initial_temp, duration))
    outputs = final_temp.reshape(-1, 1)

    return inputs, outputs


def split_data(inputs, outputs, test_ratio=0.2):
    """
    Split data into training and testing sets

    Args:
        inputs (numpy.ndarray): Input data
        outputs (numpy.ndarray): Output data
        test_ratio (float): Ratio of test data

    Returns:
        tuple: (train_inputs, train_outputs, test_inputs, test_outputs)
    """
    # Determine split index
    n_samples = len(inputs)
    split_idx = int(n_samples * (1 - test_ratio))

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Split data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_inputs = inputs[train_indices]
    train_outputs = outputs[train_indices]
    test_inputs = inputs[test_indices]
    test_outputs = outputs[test_indices]

    return train_inputs, train_outputs, test_inputs, test_outputs


def main():
    """Main function to demonstrate the ANN"""
    # Generate data
    print("Generating heat transfer data...")
    inputs, outputs = generate_heat_transfer_data(1000)

    # Split data
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)

    # Create ANN
    ann = ANN_2_2_1(learning_rate=0.05)

    # Normalize data
    ann.input_normalizer.fit(train_inputs)
    ann.output_normalizer.fit(train_outputs)

    norm_train_inputs = ann.input_normalizer.normalize(train_inputs)
    norm_train_outputs = ann.output_normalizer.normalize(train_outputs)

    norm_test_inputs = ann.input_normalizer.normalize(test_inputs)
    norm_test_outputs = ann.output_normalizer.normalize(test_outputs)

    # Set error function (choose one: 'mse', 'mae', 'bce')
    print("\nAvailable error functions:")
    print("1. Mean Squared Error (MSE)")
    print("2. Mean Absolute Error (MAE)")
    print("3. Binary Cross-Entropy (BCE)")

    error_choice = input("Select error function (1-3): ")
    if error_choice == '1':
        ann.set_error_function('mse')
    elif error_choice == '2':
        ann.set_error_function('mae')
    elif error_choice == '3':
        ann.set_error_function('bce')
    else:
        print("Invalid choice, using MSE as default")
        ann.set_error_function('mse')

    # Choose training mode
    print("\nAvailable training modes:")
    print("1. Incremental Training (weights updated after each sample)")
    print("2. Batch Training (weights updated after processing all samples)")

    training_mode = input("Select training mode (1-2): ")

    if training_mode == '1':
        print("\nTraining with incremental mode...")
        converged = ann.train_incremental(
            norm_train_inputs, norm_train_outputs,
            epochs=2000, error_threshold=0.001
        )
    else:
        print("\nTraining with batch mode...")
        converged = ann.train_batch(
            norm_train_inputs, norm_train_outputs,
            epochs=2000, error_threshold=0.001
        )

    if converged:
        # Save the model
        ann.save_model()

        # Test the model
        print("\nTesting the model on test data...")
        norm_test_outputs_pred, test_error = ann.test(norm_test_inputs, norm_test_outputs)

        # Denormalize outputs
        test_outputs_pred = ann.output_normalizer.denormalize(norm_test_outputs_pred)

        print(f"Test error: {test_error:.6f}")

        # Plot test results
        plt.figure(figsize=(10, 6))
        plt.scatter(test_outputs, test_outputs_pred, alpha=0.5)
        plt.plot([min(test_outputs), max(test_outputs)], [min(test_outputs), max(test_outputs)], 'r--')
        plt.xlabel('Actual Output')
        plt.ylabel('Predicted Output')
        plt.title('Actual vs Predicted Output')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot sample predictions
        sample_idx = np.random.choice(len(test_inputs), 5, replace=False)
        sample_inputs = test_inputs[sample_idx]
        sample_outputs = test_outputs[sample_idx]
        sample_outputs_pred = test_outputs_pred[sample_idx]

        print("\nSample predictions:")
        print("Initial Temperature (°C) | Duration (s) | Actual Final Temp (°C) | Predicted Final Temp (°C)")
        print("-" * 85)

        for i in range(len(sample_inputs)):
            print(
                f"{sample_inputs[i, 0]:20.2f} | {sample_inputs[i, 1]:11.2f} | {sample_outputs[i, 0]:22.2f} | {sample_outputs_pred[i, 0]:25.2f}")


if __name__ == "__main__":
    main()