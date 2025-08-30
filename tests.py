import numpy as np
from least_square import LeastSquares
from gradient_descent import GradientDescent
from mlp import MLP
import torch as t
import torch.nn as nn

# Test for LeastSquares class
def test_least_squares_fit_predict():
    # Test with a simple linear relationship
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([6, 8, 9, 11])
    
    model = LeastSquares()
    model.fit(X, y)
    
    # Expected weights for y = 1*x1 + 2*x2 + 3 (approx)
    # The exact weights can be calculated using the formula
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    expected_weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    np.testing.assert_array_almost_equal(model.weights, expected_weights, decimal=5)
    
    # Test prediction
    X_new = np.array([[1, 3], [3, 1]])
    predictions = model.predict(X_new)
    
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    expected_predictions = X_new_b @ expected_weights
    
    np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=5)

# Test for GradientDescent class
def test_gradient_descent_fit_predict():
    # Test with a simple linear relationship
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8]).reshape(-1, 1)

    model = GradientDescent(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    # For a perfectly linear relationship y = 2x, we expect weights to be close to [0, 2]
    # The intercept (bias) should be close to 0, and the coefficient for x should be close to 2.
    # Due to the stochastic nature of mini-batch gradient descent and random initialization,
    # we use a broader tolerance for assert_array_almost_equal.
    # Let's check if the predictions are close to the actual values.
    predictions = model.predict(X)
    np.testing.assert_array_almost_equal(predictions, y, decimal=0)

    # Test with new data
    X_new = np.array([[5], [6]])
    expected_predictions_new = np.array([10, 12]).reshape(-1, 1)
    predictions_new = model.predict(X_new)
    np.testing.assert_array_almost_equal(predictions_new, expected_predictions_new, decimal=0)

# Test for MLP class
def test_mlp_forward_pass():
    # Define input, hidden, and output sizes
    input_size = 10
    hidden_layers = [20, 30]
    output_size = 1

    # Create an instance of MLP
    model = MLP(input_size, hidden_layers, output_size)

    # Create a dummy input tensor
    X_dummy = t.randn(1, input_size)

    # Perform a forward pass
    output = model(X_dummy)

    # Check the output shape
    assert output.shape == (1, output_size)

    # Check if the output is a tensor
    assert isinstance(output, t.Tensor)

    # Test with multiple samples
    X_dummy_batch = t.randn(5, input_size)
    output_batch = model(X_dummy_batch)
    assert output_batch.shape == (5, output_size)


def test_mlp_architecture():
    # Test different architectures
    input_size = 5
    hidden_layers = [10, 15, 5]
    output_size = 2

    model = MLP(input_size, hidden_layers, output_size)

    # Check if the number of layers is as expected
    # Input layer + 3 hidden layers + output layer = 5 Linear layers
    # Each Linear layer is followed by a ReLU (except the last one)
    # So, 2*len(hidden_layers) + 1 Linear layer + (len(hidden_layers) if ReLU is after each hidden) + 1 (for first layer ReLU)
    # Let's count the Linear layers directly
    linear_layers = [module for module in model.layers if isinstance(module, nn.Linear)]
    assert len(linear_layers) == len(hidden_layers) + 1

    # Check input and output dimensions of the first and last linear layers
    assert linear_layers[0].in_features == input_size
    assert linear_layers[0].out_features == hidden_layers[0]
    assert linear_layers[-1].in_features == hidden_layers[-1]
    assert linear_layers[-1].out_features == output_size
