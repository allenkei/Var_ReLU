import math
import torch

"""
This file defines functions for generating synthetic data for simulations.
It includes functions to generate f(x) and g(x) values, as well as the response variable y.
These functions are critical for testing statistical and machine learning models in simulations.

Functions included:
1. `f_function`: Generates normalized f(x) values with range into [-1, 1].
2. `g_function`: Generates g(x) values with range into [-1, 1].
3. `gen_y`: Combines f(x) and g(x) to generate the response variable y.
"""


def f_function(q):
    """
    Generate normalized f(x) values with range into [-1, 1].
    
    Parameters:
    - q (torch.Tensor): Input tensor of shape (N, 2), where N is the number of samples.

    Returns:
    - torch.Tensor: Normalized f(x) values for each input sample.
    """
    q1, q2 = q[:, 0], q[:, 1]
    new_q1 = torch.sqrt(q1) + q1 * q2
    new_q2 = torch.cos(2 * math.pi * q2)
    scalar_output = torch.sqrt(new_q1 + new_q2**2) + new_q1**2 * new_q2
    normalized_output = scalar_output / 8
    return normalized_output


def g_function(q):
    """
    Generate g(x) values with range into [-1, 1].

    Parameters:
    - q (torch.Tensor): Input tensor of shape (N, 2), where N is the number of samples.

    Returns:
    - torch.Tensor: g(x) values for each input sample.
    """
    point = torch.tensor([0.5, 0.5])
    diff = q - point
    normalized_output = torch.norm(diff, p=2, dim=1) / 64
    return normalized_output


def gen_y(f_x, g_x, num_samples):
    """
    Generate response variable y based on f(x), g(x), and Gaussian noise.
    
    Parameters:
    - f_x (torch.Tensor): Generated f(x) values.
    - g_x (torch.Tensor): Generated g(x) values.
    - num_samples (int): Number of samples to generate.

    Returns:
    - torch.Tensor: Response variable y for each sample.
    """
    z_samples = torch.randn(num_samples)
    return f_x + torch.sqrt(g_x) * z_samples
