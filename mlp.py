import torch as t
import torch.nn as nn
import numpy as np
from typeguard import typechecked



class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for regression tasks.

    This class defines a feedforward neural network with a configurable number of
    hidden layers and neurons, using ReLU activation functions between layers.

    Attributes:
        layers (nn.Sequential): The sequential container of layers forming the MLP.
    """
    @typechecked
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int) -> None:
        """
        Initializes the MLP model.

        Args:
            input_size (int): The number of input features.
            hidden_layers (list[int]): A list where each element represents the number of neurons
                                        in a corresponding hidden layer.
            output_size (int): The number of output features.
        """
        super(MLP, self).__init__()
        
        raise NotImplementedError()

    @typechecked
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x (t.Tensor): The input tensor.

        Returns:
            t.Tensor: The output tensor from the MLP.
        """
        raise NotImplementedError()
    
@typechecked
def preprocess(X: np.ndarray, y: np.ndarray, device: str) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Preprocesses the input data by splitting it into training and testing sets,
    scaling the features, and converting them to PyTorch tensors.

    Args:
        X (np.ndarray): The input features as a NumPy array.
        y (np.ndarray): The target variable as a NumPy array.
        device (str): The device to move the tensors to (e.g., 'cpu' or 'cuda').

    Returns:
        tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]: A tuple containing:
            - X_train_tensor (t.Tensor): Scaled training features.
            - X_test_tensor (t.Tensor): Scaled testing features.
            - y_train_tensor (t.Tensor): Training target variable.
            - y_test_tensor (t.Tensor): Testing target variable.
    """
    raise NotImplementedError()

@typechecked
def setup_model(input_size: int, hidden_layers: list[int], output_size: int, device: t.device) -> tuple[MLP, nn.Module, t.optim.Optimizer]:
    """
    Sets up the MLP model, loss function, and optimizer.

    Args:
        input_size (int): The number of input features for the MLP.
        hidden_layers (list[int]): A list defining the architecture of hidden layers.
        output_size (int): The number of output features for the MLP.
        device (t.device): The device (CPU or GPU) to deploy the model to.

    Returns:
        tuple[MLP, nn.Module, t.optim.Optimizer]: A tuple containing:
            - model (MLP): The initialized MLP model.
            - criterion (nn.Module): The loss function.
            - optimizer (t.optim.Optimizer): The optimizer for updating model parameters.
    """
    raise NotImplementedError()

@typechecked
def train_model(model: MLP, train_loader: t.utils.data.DataLoader, criterion: nn.Module, optimizer: t.optim.Optimizer, num_epochs: int) -> MLP:
    """
    Trains the MLP model using the provided data loader, criterion, and optimizer.

    Args:
        model (MLP): The MLP model to be trained.
        train_loader (t.utils.data.DataLoader): The data loader providing training batches.
        criterion (nn.Module): The loss function.
        optimizer (t.optim.Optimizer): The optimizer for updating model parameters.
        num_epochs (int): The number of training epochs.

    Returns:
        MLP: The trained MLP model.
    """
    raise NotImplementedError()
