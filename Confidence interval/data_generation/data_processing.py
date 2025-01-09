"""
This file contains functions to generate and partition synthetic datasets for each Monte Carlo repetition.

Functions included:
1. `generate_data`: Generates synthetic data to be used for the coverage for each Monte Carlo trials.
2. `partition_dataset`: Partitions the generated dataset into subsets I1, I2, I3, and I4.
"""


import numpy as np
import torch
from data_generation.data_functions import f_function, g_function, gen_y


def generate_data(num_samples, confidence_rep_quantity, trial_out, input_dim):
    """
    Generate synthetic data for each Monte Carlo trial.

    Parameters:
    - num_samples (int): Number of samples to generate for each trial.
    - confidence_rep_quantity (int): Number of datasets to be generated to compute the coverage probability. Usually 100 of them are need to compute such coverage.
    - trial_out (int): Identifier for the outer trial (used for seed generation), representing the corresponding monte carlo repetition.
    - input_dim (int): Dimensionality of input features.

    Returns:
    - all_data (list): List of tensors, where each tensor contains data for one trial among the 100 datasets needed to compute such coverage, in a fixed monte carlo repetition.
                       Each tensor has columns for f(x), g(x), y, and input features.
    - An_values (list): List of values for each trial, used as A_n thresholds.
    """
    all_data = []
    An_values = []
    for i in range(confidence_rep_quantity):
        torch.manual_seed(i + 100 * trial_out)
        np.random.seed(i + 100 * trial_out)
        x_samples = torch.rand(num_samples, input_dim)
        f_values = f_function(x_samples)
        g_values = g_function(x_samples)
        y_samples = gen_y(f_values, g_values, num_samples)
        trial_data = torch.cat((
            torch.tensor(f_values).view(-1, 1),
            torch.tensor(g_values).view(-1, 1),
            torch.tensor(y_samples).view(-1, 1),
            x_samples
        ), dim=1)
        An_values.append(max(y_samples))
        all_data.append(trial_data)
    return all_data, An_values  # List of tensors, one for each trial


def partition_dataset(all_data, num_samples):
    """
    Partition the dataset into subsets I1, I2, I3, and I4 for each trial, among the 100 needed to compute the coverage in a fixed monte carlo repetition.

    Parameters:
    - all_data (list): List of tensors, where each tensor contains data for one trial.
                       Each tensor has columns for f(x), g(x), y, and input features.
    - num_samples (int): Number of samples in each trial.

    Returns:
    - partitions (dict): Dictionary with keys 'I1', 'I2', 'I3', and 'I4'.
                         Each key maps to a list of tensors (one per trial) containing the corresponding partition.
    """
    I1_partitions, I2_partitions, I3_partitions, I4_partitions = [], [], [], []

    for trial_data in all_data:
        indices = torch.randperm(num_samples)
        partition_size = num_samples // 4
        I1 = trial_data[indices[:partition_size]]
        I2 = trial_data[indices[partition_size:2 * partition_size]]
        I3 = trial_data[indices[2 * partition_size:3 * partition_size]]
        I4 = trial_data[indices[3 * partition_size:]]
        I1_partitions.append(I1)
        I2_partitions.append(I2)
        I3_partitions.append(I3)
        I4_partitions.append(I4)

    return {'I1': I1_partitions, 'I2': I2_partitions, 'I3': I3_partitions, 'I4': I4_partitions}
