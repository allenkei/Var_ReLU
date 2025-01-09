"""
This module computes normalized residuals and their empirical distributions 
for the I2 partition of the dataset for each trial.
"""

import torch
from torch.nn.functional import mse_loss

def compute_normalized_residuals_and_empirical_distribution_with_An(
    partitions, f_models, g_models, An_values, confidence_rep_quantity, device
):
    """
    Compute normalized residuals for the I2 partition in each trial and construct empirical distribution F^(t),
    incorporating A_n.

    Parameters:
    - partitions (dict): Dictionary containing partitions (I1, I2, I3, I4) for all trials.
    - f_models (list): List of trained f^ models from Step 2.
    - g_models (list): List of trained g^ models from Step 3.
    - An_values (list): List of computed A_n values for each trial.
    - confidence_rep_quantity (int): Number of Monte Carlo trials.
    - device (torch.device): Torch device (CPU or GPU) to use for computations.

    Returns:
    - normalized_residuals (list): A list of tensors, one for each trial, containing normalized residuals for I2.
    - empirical_distributions (list): A list of empirical CDFs (as tuples of sorted residuals and CDF values) for each trial.
    """
    normalized_residuals = []  # To store the normalized residuals for each trial
    empirical_distributions = []  # To store the empirical distributions for each trial

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Computing normalized residuals and empirical distribution with A_n for trial {trial + 1}/{confidence_rep_quantity}...")

        # Get the I2 partition for this trial
        I2 = partitions['I2'][trial]
        x_samples = I2[:, 3:].to(device)  # Covariates (x1, x2, ...)
        y_samples = I2[:, 2].to(device)   # Responses (y)
        An = An_values[trial]  # A_n for this trial

        # Get the trained models for this trial
        model_f = f_models[trial]
        model_g = g_models[trial]

        with torch.no_grad():
            # Compute f_predictions and clamp them
            f_predictions = model_f(x_samples).clone().squeeze()
            f_predictions_clamped = f_predictions.clamp(-An, An)
            residuals = y_samples - f_predictions_clamped  # Compute residuals

            # Compute g_predictions and clamp them
            g_predictions = model_g(x_samples).clone().squeeze().clamp(-An, An)
            normalized = residuals / torch.sqrt(torch.abs(g_predictions))  # Normalize residuals

        # Standardize the normalized residuals
        mean_normalized = normalized.mean()
        std_normalized = normalized.std()
        standardized_normalized = (normalized - mean_normalized) / std_normalized

        # Store normalized residuals for this trial
        normalized_residuals.append(standardized_normalized)

        # Compute empirical CDF
        sorted_residuals = torch.sort(standardized_normalized).values
        cdf_values = torch.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
        empirical_distributions.append((sorted_residuals, cdf_values))

    return normalized_residuals, empirical_distributions
