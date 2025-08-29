import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typeguard import typechecked

class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float=0.001, n_iterations: int=1000, batch_size: int=32) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples = X_b.shape[0]
        n_features = X_b.shape[1]
        self.weights = np.random.randn(n_features, 1)

        for iteration in range(self.n_iterations):
            shuffled_indices = np.random.permutation(n_samples)
            X_b_shuffled = X_b[shuffled_indices]
            y_shuffled = y[shuffled_indices].reshape(-1, 1)

            for i in range(0, n_samples, self.batch_size):
                xi = X_b_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                
                gradients = 2/self.batch_size * xi.T @ (xi @ self.weights - yi)
                self.weights = self.weights - self.learning_rate * gradients
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
