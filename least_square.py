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
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

@typechecked
def preprocess(dataset: pd.DataFrame, subsample_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
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
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test