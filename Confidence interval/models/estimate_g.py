"""
This script trains g^ models using the I2 partition of the dataset for each dataset generated to compute coverage in a fixed monte carlo repetition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from models.nn_model import NN


def estimate_g_I2_with_An(partitions, f_models, confidence_rep_quantity, An, input_dim, hidden_dim_g, output_dim, lr2, num_epochs, device):
    
    """
    trial=a dataset generated to compute coverage, in a fixed monte carlo repetition.
    Estimate g^ using the I2 partition for each trial, incorporating A_n.

    Parameters:
    - partitions (dict): Dictionary containing partitions (I1, I2, I3, I4) for all trials.
    - f_models (list): List of trained f^ models from Step 2.
    - confidence_rep_quantity (int): Number of Monte Carlo trials.
    - An (list): List of computed A_n values for each trial.
    - input_dim (int): Dimensionality of input data.
    - hidden_dim_g (int): Hidden layer size for the g^ model.
    - output_dim (int): Output dimension of the g^ model.
    - lr2 (float): Learning rate for the optimizer.
    - num_epochs (int): Number of training epochs.
    - device (torch.device): Torch device (CPU or GPU) to use for computations.

    Returns:
    - g_models (list): A list of trained g^ models (one for each trial).
    """
    g_models = []  # To store the trained models for each trial
    criterion = nn.MSELoss()  # Mean squared error loss

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Estimating g^ for dataset number {trial + 1}/{confidence_rep_quantity} among the {confidence_rep_quantity} needed to compute coverage in a fixed monte carlo repetition...")


        # Get the I2 partition for this trial
        I2 = partitions['I2'][trial]
        x_samples = I2[:, 3:].to(device)  # Covariates (x1, x2, ...)
        y_samples = I2[:, 2].to(device)   # Responses (y)

        # Get the trained f^ model for this trial
        model_f = f_models[trial]

        with torch.no_grad():
            f_predictions = model_f(x_samples).clone().squeeze()
            f_predictions_clamped = f_predictions.clamp(-An[trial], An[trial])  # Clamp f_predictions to [-A_n, A_n]
            residuals_squared = (y_samples - f_predictions_clamped) ** 2  # Compute residuals squared

        # Initialize the neural network for g^
        model_g = NN(input_dim, hidden_dim_g, output_dim).to(device)
        optimizer = optim.Adam(model_g.parameters(), lr=lr2)

        # Train the model to minimize MSE on residuals
        for epoch in range(num_epochs):
            outputs = model_g(x_samples)
            loss = criterion(outputs.squeeze(), residuals_squared)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Append the trained model to the list
        g_models.append(model_g)

    return g_models