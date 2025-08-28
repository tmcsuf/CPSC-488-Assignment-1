import torch as t
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, output_size: int) -> None:
        super(MLP, self).__init__()
        
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

def preprocess(X: np.ndarray, y: np.ndarray, device: str) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = t.tensor(X_train_scaled, dtype=t.float32).to(device)
    y_train_tensor = t.tensor(y_train, dtype=t.float32).unsqueeze(1).to(device)
    X_test_tensor = t.tensor(X_test_scaled, dtype=t.float32).to(device)
    y_test_tensor = t.tensor(y_test, dtype=t.float32).unsqueeze(1).to(device)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def setup_model(input_size: int, hidden_layers: list[int], output_size: int, device: t.device) -> tuple[MLP, nn.Module, t.optim.Optimizer]:
    model = MLP(input_size, hidden_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

def train_model(model: MLP, train_loader: t.utils.data.DataLoader, criterion: nn.Module, optimizer: t.optim.Optimizer, num_epochs: int) -> MLP:
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model: MLP, criterion: nn.Module, X_test_tensor: t.Tensor, y_test_tensor: t.Tensor) -> float:
    model.eval()
    with t.no_grad():
        outputs = model(X_test_tensor)
        mse = criterion(outputs, y_test_tensor)
        return mse
