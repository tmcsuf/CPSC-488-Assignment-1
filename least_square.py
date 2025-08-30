import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typeguard import typechecked

class LeastSquares:
    @typechecked
    def __init__(self) -> None:
        self.weights = None
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using the least squares method.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

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
    # Apply subsampling
    if subsample_ratio < 1.0:
        dataset = dataset.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
        print(f'Subsampled dataset to {len(dataset)} rows ({subsample_ratio*100:.0f}% of original)')

    # Separate features (X) and target (y)
    X = dataset[['T', 'P', 'TC', 'SV']].values
    y = dataset['Idx'].values

    return X, y

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
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test