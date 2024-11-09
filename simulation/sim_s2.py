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

  parser.add_argument('--scenario', default='s2')
  parser.add_argument('--num_samples', default=100, type=int)
  parser.add_argument('--num_trials', default=100, type=int)
  parser.add_argument('--input_dim', default=2, type=int)
  parser.add_argument('--output_dir', default='./data/')
  parser.add_argument('-f', required=False) # needed in Colab 

  return parser.parse_args()


args = parse_args(); print(args)
os.makedirs(args.output_dir, exist_ok=True)



def f_function(q):
    q1, q2 = q[:, 0], q[:, 1]
    
    new_q1 = torch.log(1 + q1**2) + q1 * q2
    new_q2 = torch.sin(3 * math.pi * q2) * torch.exp(-q1)
    scalar_output = (new_q1**2 + new_q2**2)**(1/3) + new_q1 / (1 + torch.abs(new_q2))
    return scalar_output

def g_function(q):
    point = torch.tensor([0.25, 0.75])
    diff = q - point
    return torch.norm(diff, p=2, dim=1)

def gen_y(f_x, g_x, num_samples):
    z_samples = torch.randn(num_samples)  # Normal(0,1)
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

    #if i <= 2:
    #  print(x_samples[0,:], f_values[0], g_values[0], y_samples[0])
    
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
df_all_data = pd.DataFrame(all_data_combined, columns=["trial", "f_value", "g_value", "y_value", "x1", "x2"])
df_all_data.to_csv(os.path.join(args.output_dir, f"{args.scenario}_data_n{args.num_samples}.csv"), index=False)
print(f'[INFO] {args.scenario} simulated data saved')





















