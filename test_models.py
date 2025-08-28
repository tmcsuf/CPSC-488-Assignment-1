import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch as t
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from least_square import LeastSquares
from gradient_descent import GradientDescent
from mlp import MLP, preprocess as mlp_preprocess, setup_model, train_model, evaluate_model
import least_square
import gradient_descent

def add_polynomial_features(x: np.ndarray, order: int) -> np.ndarray:
    if order < 2:
        return x
    x_poly = x.copy()
    for i in range(2, order + 1):
        x_poly = np.c_[x_poly, x**i]
    return x_poly

def run_tests():
    results = []
    dataset = pd.read_csv('GasProperties.csv')
    subsample_ratio = 0.4

    # Apply subsampling to the dataset once
    if subsample_ratio < 1.0:
        dataset = dataset.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
        print(f'Subsampled dataset to {len(dataset)} rows ({subsample_ratio*100:.0f}% of original)')

    for degree in range(1, 6):
        print(f"\n--- Running tests with polynomial degree {degree} ---")

        X_raw = dataset[['T', 'P', 'TC', 'SV']].values
        y_raw = dataset['Idx'].values

        # --- Least Squares ---
        print("Running Least Squares...")
        X_ls_poly = add_polynomial_features(X_raw, degree)
        X_train_ls, X_test_ls, y_train_ls, y_test_ls = least_square.scale_and_split(X_ls_poly, y_raw)
        
        
        model_ls = LeastSquares()

        start_time_ls = time.time()
        model_ls.fit(X_train_ls, y_train_ls)
        end_time_ls = time.time()

        training_time_ls = end_time_ls - start_time_ls

        y_train_pred_ls = model_ls.predict(X_train_ls)
        y_test_pred_ls = model_ls.predict(X_test_ls)

        rmse_train_ls = np.sqrt(mean_squared_error(y_train_ls, y_train_pred_ls))
        r2_train_ls = r2_score(y_train_ls, y_train_pred_ls)
        rmse_test_ls = np.sqrt(mean_squared_error(y_test_ls, y_test_pred_ls))
        r2_test_ls = r2_score(y_test_ls, y_test_pred_ls)

        results.append({
            'Model': 'Least Squares',
            'Degree': degree,
            'Train RMSE': rmse_train_ls,
            'Train R^2': r2_train_ls,
            'Test RMSE': rmse_test_ls,
            'Test R^2': r2_test_ls,
            'Training Time (s)': training_time_ls
        })

        # --- Gradient Descent ---
        print("Running Gradient Descent...")
        X_gd_poly = add_polynomial_features(X_raw, degree)
        X_train_gd, X_test_gd, y_train_gd, y_test_gd = gradient_descent.scale_and_split(X_gd_poly, y_raw)

        model_gd = GradientDescent(n_iterations=1000, learning_rate=0.001)

        start_time_gd = time.time()
        model_gd.fit(X_train_gd, y_train_gd.reshape(-1, 1))
        end_time_gd = time.time()

        training_time_gd = end_time_gd - start_time_gd

        y_train_pred_gd = model_gd.predict(X_train_gd)
        y_test_pred_gd = model_gd.predict(X_test_gd)

        rmse_train_gd = np.sqrt(mean_squared_error(y_train_gd, y_train_pred_gd))
        r2_train_gd = r2_score(y_train_gd, y_train_pred_gd)
        rmse_test_gd = np.sqrt(mean_squared_error(y_test_gd, y_test_pred_gd))
        r2_test_gd = r2_score(y_test_gd, y_test_pred_gd)

        results.append({
            'Model': 'Gradient Descent',
            'Degree': degree,
            'Train RMSE': rmse_train_gd,
            'Train R^2': r2_train_gd,
            'Test RMSE': rmse_test_gd,
            'Test R^2': r2_test_gd,
            'Training Time (s)': training_time_gd
        })

        # --- MLP ---
        print("Running MLP...")
        device = t.device('cuda' if t.cuda.is_available() else 'cpu') # Revert to checking for CUDA availability

        # Add polynomial features before preprocessing for MLP
        X_mlp_poly_np = add_polynomial_features(X_raw, degree)
        
        # Preprocess MLP data using its specific preprocess function
        X_train_mlp_tensor, y_train_mlp_tensor, X_test_mlp_tensor, y_test_mlp_tensor = mlp_preprocess(X_mlp_poly_np, y_raw, device)

        input_size_mlp = X_train_mlp_tensor.shape[1]
        hidden_layers_mlp = [64, 32] 
        output_size_mlp = 1 
        num_epochs_mlp = 100
        batch_size_mlp = 64

        model_mlp, criterion_mlp, optimizer_mlp = setup_model(input_size_mlp, hidden_layers_mlp, output_size_mlp, device)
        train_dataset_mlp = TensorDataset(X_train_mlp_tensor, y_train_mlp_tensor)
        train_loader_mlp = DataLoader(dataset=train_dataset_mlp, batch_size=batch_size_mlp, shuffle=True)

        start_time_mlp = time.time()
        model_mlp = train_model(model_mlp, train_loader_mlp, criterion_mlp, optimizer_mlp, num_epochs_mlp)
        end_time_mlp = time.time()

        training_time_mlp = end_time_mlp - start_time_mlp

        model_mlp.eval()
        with t.no_grad():
            y_train_pred_mlp = model_mlp(X_train_mlp_tensor).cpu().numpy()
            y_test_pred_mlp = model_mlp(X_test_mlp_tensor).cpu().numpy()

        rmse_train_mlp = np.sqrt(mean_squared_error(y_train_mlp_tensor.cpu().numpy(), y_train_pred_mlp))
        r2_train_mlp = r2_score(y_train_mlp_tensor.cpu().numpy(), y_train_pred_mlp)
        rmse_test_mlp = np.sqrt(mean_squared_error(y_test_mlp_tensor.cpu().numpy(), y_test_pred_mlp))
        r2_test_mlp = r2_score(y_test_mlp_tensor.cpu().numpy(), y_test_pred_mlp)

        results.append({
            'Model': 'MLP',
            'Degree': degree,
            'Train RMSE': rmse_train_mlp,
            'Train R^2': r2_train_mlp,
            'Test RMSE': rmse_test_mlp,
            'Test R^2': r2_test_mlp,
            'Training Time (s)': training_time_mlp
        })
    
    # Display results in a table
    results_df = pd.DataFrame(results)
    print("\n--- Test Results ---")
    print(results_df.to_string())

if __name__ == "__main__":
    run_tests()
