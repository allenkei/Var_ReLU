import os
import math
import numpy as np
import pandas as pd
import argparse
import torch
import random

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--scenario', default='s5')
  parser.add_argument('--num_samples', default=100, type=int)
  parser.add_argument('--num_trials', default=100, type=int)
  parser.add_argument('--input_dim', default=5, type=int)
  parser.add_argument('--output_dir', default='./data/')
  parser.add_argument('-f', required=False) # needed in Colab 

  return parser.parse_args()


args = parse_args(); print(args)
os.makedirs(args.output_dir, exist_ok=True)



def f_function(q):

    q1, q2, q3, q4, q5 = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4]
    
    new_q1 = q1**2 + q2**2 + q3 + q4 + q5
    new_q2 = q1 + q2 + q3**2 + q4**2 + q5**2
    
    scalar_output = new_q1 + new_q2 + new_q1 * new_q2
    
    return scalar_output

def g_function(q):

    point = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    diff = q - point
    
    norm_1 = torch.norm(diff, p=2, dim=1)
    norm_2 = torch.sum(torch.abs(diff), dim=1)
    
    scalar_output = norm_1 + torch.sqrt(norm_2)
    
    return scalar_output
    

def gen_y(f_x, g_x, num_samples):

    low = -3**0.5  # Lower bound
    high = 3**0.5  # Upper bound
    uniform_dist = torch.distributions.Uniform(low, high)
    z_samples = uniform_dist.sample((num_samples,))

    return f_x + torch.sqrt(g_x) * z_samples








all_data = []





for i in range(args.num_trials):

    #################
    # Generate Data #
    #################
    x_samples = torch.rand(args.num_samples, args.input_dim) # Unif(0,1)
    f_values = f_function(x_samples)
    g_values = g_function(x_samples)
    y_samples = gen_y(f_values, g_values, args.num_samples) # use both f and g
    
    x_samples_np = x_samples.numpy()
    f_values_np = f_values.numpy().reshape(-1, 1)
    g_values_np = g_values.numpy().reshape(-1, 1)
    y_samples_np = y_samples.numpy().reshape(-1, 1)

    #if i <= 2:
    #  print(x_samples_np[0,:], f_values_np[0], g_values_np[0], y_samples_np[0])

    trial_col = np.full((args.num_samples, 1), i + 1)
    trial_data = np.hstack((trial_col, f_values_np, g_values_np, y_samples_np, x_samples_np))
    all_data.append(trial_data)

    #if i <= 2:
    #  print(all_data[i][0])

all_data_combined = np.vstack(all_data)

base_columns = ["trial", "f_value", "g_value", "y_value"]
x_columns = [f"x{i}" for i in range(1, args.input_dim + 1)]

all_columns = base_columns + x_columns
df_all_data = pd.DataFrame(all_data_combined, columns=all_columns)
df_all_data.to_csv(os.path.join(args.output_dir, f"{args.scenario}_data_n{args.num_samples}.csv"), index=False)
print(f'[INFO] {args.scenario} simulated data saved')





















