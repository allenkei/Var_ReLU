"""
This script trains f^ models using the I1 partition of the dataset for each dataset generated to compute coverage in a fixed monte carlo repetition.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.nn_model import NN


def estimate_f_I1(partitions, confidence_rep_quantity, input_dim, hidden_dim_f, output_dim, lr1, num_epochs, device):
    """
    trial=a dataset generated to compute coverage, in a fixed monte carlo repetition.
    Estimate f^ using the I1 partition for each trial.

    Parameters:
    - partitions (dict): Dictionary containing partitions (I1, I2, I3, I4) for all trials.
    - confidence_rep_quantity (int): Number of Monte Carlo trials.
    - input_dim (int): Dimensionality of input data.
    - hidden_dim_f (int): Hidden layer size for the f^ model.
    - output_dim (int): Output dimension of the f^ model.
    - lr1 (float): Learning rate for the optimizer.
    - num_epochs (int): Number of training epochs.
    - device (torch.device): Torch device (CPU or GPU) to use for computations.

    Returns:
    - f_models (list): A list of trained f^ models (one for each trial).
    """
    f_models = []  # To store the trained models for each trial
    criterion = nn.MSELoss()  # Mean squared error loss

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Estimating f^ for dataset number {trial + 1}/{confidence_rep_quantity} among the {confidence_rep_quantity} needed to compute coverage in a fixed monte carlo repetition...")

        # Get the I1 partition for this trial
        I1 = partitions['I1'][trial]
        x_samples = I1[:, 3:].to(device)  # Covariates (x1, x2, ...)
        y_samples = I1[:, 2].to(device)   # Responses (y)

        # Initialize the neural network for this trial
        model_f = NN(input_dim, hidden_dim_f, output_dim).to(device)
        optimizer = optim.Adam(model_f.parameters(), lr=lr1)

        # Train the model to minimize MSE on I1
        for epoch in range(num_epochs):
            outputs = model_f(x_samples)
            loss = criterion(outputs.squeeze(), y_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f_models.append(model_f)

    return f_models