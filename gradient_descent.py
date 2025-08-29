import pandas as pd
import numpy as np
from typeguard import typechecked

class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float, n_iterations: int, batch_size: int) -> None:
        raise NotImplementedError()
        
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using mini-batch Gradient Descent.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """
        raise NotImplementedError()
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        raise NotImplementedError()
    
@typechecked
def preprocess(dataset: pd.DataFrame, subsample_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the input dataset by optionally subsampling it and separating
    features (X) and target (y).

    Args:
        dataset (pd.DataFrame): The input dataset containing features and target.
        subsample_ratio (float, optional): The ratio of the dataset to subsample.
                                           Defaults to 1.0 (no subsampling).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): The processed feature matrix.
            - y (np.ndarray): The processed target vector.
    """
    raise NotImplementedError()

@typechecked
def scale_and_split(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales the input features and splits the data into training and testing sets.

    Args:
        X (np.ndarray): The input features as a NumPy array.
        y (np.ndarray): The target variable as a NumPy array.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_train (np.ndarray): Scaled training features.
            - X_test (np.ndarray): Scaled testing features.
            - y_train (np.ndarray): Training target variable.
            - y_test (np.ndarray): Testing target variable.
    """
    raise NotImplementedError()
