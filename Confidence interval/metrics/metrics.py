"""
This module contains functions for computing coverage metrics, 
constructing confidence intervals, and estimating parameters a(alpha) and b(alpha).
"""

import numpy as np
import math
import torch


def compute_delta(alpha, a_alpha, b_alpha, B_tilde):
    """
    Compute Delta(alpha) based on the formula:
    Delta(alpha) = 2 * sqrt(a(alpha) / (alpha * B_tilde)) + b(alpha)

    Parameters:
    - alpha (float): Significance level
    - a_alpha (float): Precomputed a(alpha) value
    - b_alpha (float): Precomputed b(alpha) value
    - B_tilde (int): Number of bootstrap iterations smaller than B

    Returns:
    - delta_alpha (float): Computed Delta(alpha) value
    """
    delta_alpha = 2 * math.sqrt(max(a_alpha / (alpha * B_tilde), 0)) + b_alpha
    return delta_alpha


def construct_confidence_interval(a_alpha_values, b_alpha_values, confidence_rep_quantity,alpha,B_tilde):
    """
    Construct confidence intervals for each trial.

    Parameters:
    - a_alpha_values (list): List of a(alpha) values for each trial
    - b_alpha_values (list): List of b(alpha) values for each trial
    - confidence_rep_quantity (int): Number of Monte Carlo trials
    - alpha (float): Significance level
    - B_tilde (int): Number of bootstrap iterations smaller than B

    Returns:
    - confidence_intervals (list): List of confidence intervals (tuples) for each trial
    """
    confidence_intervals = []

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Constructing confidence interval for dataset number {trial + 1}/{confidence_rep_quantity} among the {confidence_rep_quantity} needed to compute coverage in a fixed monte carlo repetition...")
        # Compute Delta(alpha)
        a_alpha = a_alpha_values[trial]
        b_alpha = b_alpha_values[trial]
        delta_alpha = compute_delta(alpha, a_alpha, b_alpha,B_tilde)
        lower_bound = -delta_alpha
        upper_bound =  delta_alpha
        confidence_intervals.append((lower_bound, upper_bound))
    return confidence_intervals


def compute_a1_alpha(partitions, bootstrap_f_models, An_values, confidence_rep_quantity, B, alpha,device):
    """
    Compute a1(alpha) for each trial.

    Parameters:
    - partitions (dict): Dictionary containing partitions for all trials
    - bootstrap_f_models (list): List of lists of trained f^ models for each bootstrap iteration
    - An_values (list): List of computed A_n values for each trial
    - confidence_rep_quantity (int): Number of Monte Carlo trials
    - B (int): Number of bootstrap iterations
    - alpha (float): Significance level
    - device (torch.device): PyTorch device to use for computation

    Returns:
    - a1_alpha_values (list): List of a1(alpha) values for each trial
    """
    a1_alpha_values = []  # To store a1(alpha) for each trial

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Computing a1(alpha) for dataset number {trial + 1}/{confidence_rep_quantity} among the {confidence_rep_quantity} needed to compute coverage in a fixed monte carlo repetition...")
        
        I4 = partitions['I4'][trial]
        x_samples_I4 = I4[:, 3:].to(device)  # Covariates (x1, x2, ...)
        y_samples_I4 = I4[:, 2].to(device)   # Responses (y)
        An = An_values[trial]  # Get A_n for this trial

        # Compute errors for each bootstrap model
        prediction_errors = []
        for j in range(B + 1):
            model_f_bootstrap = bootstrap_f_models[trial][j]
            with torch.no_grad():
                # Clamp predictions to [-A_n, A_n]
                predictions = model_f_bootstrap(x_samples_I4).clone().squeeze().clamp(-An, An)
                mse = torch.mean((y_samples_I4 - predictions) ** 2).item()  # Mean squared error
            prediction_errors.append(mse)

        # Compute the (1 - alpha * kappa_n / 4)-quantile of prediction errors
        quantile_level = 1 - alpha/4
        a1_alpha = np.quantile(prediction_errors, quantile_level)
        a1_alpha_values.append(a1_alpha)

    return a1_alpha_values


def compute_a_and_b_alpha(partitions, bootstrap_g_models, An_values, a1_alpha_values, confidence_rep_quantity, alpha, a_0,sample_size,B_tilde,device,num_samples):
    """
    Compute a(alpha) and b(alpha) for each trial.

    Parameters:
    - partitions (dict): Dictionary containing partitions for all trials
    - bootstrap_g_models (list): List of trained g^ models for each bootstrap iteration
    - An_values (list): List of computed A_n values for each trial
    - a1_alpha_values (list): List of a1(alpha) values for each trial
    - confidence_rep_quantity (int): Number of Monte Carlo trials
    - alpha (float): Significance level
    - a_0 (float): Additional tuning parameter
    - sample_size (int): Number of samples in the dataset
    - B_tilde (int): Number of bootstrap iterations smaller than B
    - device (torch.device): PyTorch device to use for computation
    - num_samples (int): Total number of samples

    Returns:
    - a_alpha_values (list): List of a(alpha) values for each trial
    - b_alpha_values (list): List of b(alpha) values for each trial
    """
    a_alpha_values = []  # To store a(alpha) for each trial
    b_alpha_values = []  # To store b(alpha) for each trial

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Computing a(alpha) and b(alpha) for dataset number {trial + 1}/{confidence_rep_quantity} among the {confidence_rep_quantity} needed to compute coverage in a fixed monte carlo repetition...")
        
        
        # Extract trial-specific values
        I4 = partitions['I4'][trial]
        x_samples_I4 = I4[:, 3:].to(device)  # Covariates (x1, x2, ...)
        #y_samples_I4 = I4[:, 2].to(device)   # Responses (y)
        An = An_values[trial]

        # Extract f^(B+1) and g^(B+1) models
        #model_f_B_plus_1 = bootstrap_f_models[trial][-1]  # Last f^ model
        model_g_B_plus_1 = bootstrap_g_models[trial][-1]  # Last g^ model


        #I1 = partitions['I1'][trial]
        #y_samples_I1 = I1[:, 2].to(device)
        #I2 = partitions['I2'][trial]
        #y_samples_I2 = I2[:, 2].to(device)
        #I3 = partitions['I3'][trial]
        #y_samples_I3 = I3[:, 2].to(device)
        with torch.no_grad():
            # Compute predictions for I4 using f^(B+1) and g^(B+1)
            #f_predictions_B_plus_1 = model_f_B_plus_1(x_samples_I4).clone().squeeze().clamp(-An, An)
            g_predictions_B_plus_1 = model_g_B_plus_1(x_samples_I4).clone().squeeze().clamp(-An, An)

            # Mean of y_samples_I4 (y-bar)
            # Concatenate all the vectors
            #y_samples_concat = torch.cat([y_samples_I1, y_samples_I2, y_samples_I3, y_samples_I4])

            # Compute the mean of the concatenated vector
            #y_bar = y_samples_concat.mean()
            #y_bar = (y_samples_I4+y_samples_I3+y_samples_I2+y_samples_I1).mean().item()

            # Compute the mean prediction of g^(B+1)
            g_mean_B_plus_1 = g_predictions_B_plus_1.mean().item()

            # Compute the mean prediction of f^(B+1)
            #f_mean_B_plus_1 = f_predictions_B_plus_1.mean().item()

        # Calculate a(alpha)
        a1_alpha = a1_alpha_values[trial]  # Use precomputed a1(alpha)
        a_alpha = abs(
    a1_alpha
    + (An**2)*math.sqrt((32 * math.log(8 / (alpha))+ math.log(B_tilde)) / sample_size)
    #+ (16 * An**2) / (math.sqrt(sample_size * 1 * alpha))
    + (An)*math.sqrt((8 * math.log(64 / (alpha ))+ math.log(B_tilde)) / sample_size)
    - g_mean_B_plus_1
    + a_0
)
        a_alpha_values.append(a_alpha)
        b_alpha = 1/ (100 * np.log(num_samples) ** 2)
        b_alpha_values.append(b_alpha)
    return a_alpha_values, b_alpha_values