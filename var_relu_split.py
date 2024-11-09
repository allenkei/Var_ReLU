import os
import math
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(1)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--num_samples', default=500, type=int)
  parser.add_argument('--scenario', default='s1', type=str)
  parser.add_argument('--input_dim', default=2, type=int)
  parser.add_argument('--hidden_dim', default=64, type=int)
  parser.add_argument('--output_dim', default=1) 
  parser.add_argument('--lr1', default=0.001) # M1 estimator based on residuals
  parser.add_argument('--lr2', default=0.001) # M1 estimator based on residuals
  parser.add_argument('--lr3', default=0.001) # M2 direct estimator
  parser.add_argument('--num_epochs', default=1000)
  parser.add_argument('--num_trials', default=100, type=int)
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('--output_dir', default='./result_split/')
  parser.add_argument('-f', required=False) # needed in Colab 

  return parser.parse_args()


args = parse_args(); print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)
os.makedirs(args.output_dir, exist_ok=True)


df_all_data = pd.read_csv(os.path.join(args.data_dir,f"{args.scenario}_data_n{args.num_samples}.csv"))



x_columns = [f"x{j}" for j in range(1, args.input_dim + 1)]
print('[INFO] columns:', x_columns)




class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, out):
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out




MSE_g_M1_holder = torch.zeros(args.num_trials) # estimator based on residuals
MSE_g_M2_holder = torch.zeros(args.num_trials) # direct estimator
MSE_g_M3_holder = np.zeros(args.num_trials) # RF - estimator based on residuals
MSE_g_M4_holder = np.zeros(args.num_trials) # RF - direct estimator
criterion = nn.MSELoss()






for i in range(args.num_trials):
    
    trial_data = df_all_data[df_all_data['trial'] == (i+1)]
    x_samples = trial_data[x_columns].values     # Shape: (num_samples, input_dim)
    #f_values = trial_data['f_value'].values      # Shape: (num_samples,)
    g_values = trial_data['g_value'].values      # Shape: (num_samples,)
    y_samples = trial_data['y_value'].values     # Shape: (num_samples,)
    
    x_samples = torch.tensor(x_samples, dtype=torch.float32).to(device)
    #f_values = torch.tensor(f_values, dtype=torch.float32).to(device)
    g_values = torch.tensor(g_values, dtype=torch.float32).to(device)
    y_samples = torch.tensor(y_samples, dtype=torch.float32).to(device)
    

    mid_idx = len(x_samples) // 2
    
    # Split the data for 'first' half and 'second' half
    x_samples_first = x_samples[:mid_idx]
    #f_values_first = f_values[:mid_idx]
    g_values_first = g_values[:mid_idx]
    y_samples_first = y_samples[:mid_idx]
    
    x_samples_second = x_samples[mid_idx:]
    #f_values_second = f_values[mid_idx:]
    g_values_second = g_values[mid_idx:]
    y_samples_second = y_samples[mid_idx:]

    ####################
    # Neural Network 1 #
    ####################

    model1 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr1)


    # Training the model1
    for epoch in range(args.num_epochs):

        outputs1 = model1(x_samples_first) # FIRST
        loss1 = criterion(outputs1.squeeze(), y_samples_first) # FIRST
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        '''
        if (epoch+1) % 100 == 0:
            print(f'Trial {i} - NN1 - Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss1.item():.4f}')
        '''

    #####################
    # Residuals squared #
    #####################

    with torch.no_grad():
      prediction1 = model1(x_samples_second).clone() # SECOND
      res1 = y_samples_second - prediction1.squeeze() # SECOND
      res1_squared = res1 ** 2

    ####################
    # Neural Network 2 #
    ####################

    model2 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr2)

    # Train the model2
    for epoch in range(args.num_epochs):

        outputs2 = model2(x_samples_second) # SECOND
        loss2 = criterion(outputs2.squeeze(), res1_squared) # NOTE: USE residual squared # SECOND
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        '''
        if (epoch+1) % 100 == 0:
            print(f'Trial {i+1} - NN2 - Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss2.item():.4f}')
        '''

    with torch.no_grad():
      prediction2 = model2(x_samples_second).clone() # SECOND



    MSE_g_M1_holder[i] = torch.mean( (prediction2.squeeze() - g_values_second)**2 ) # SECOND



    #########################################
    # Neural Network 3 for Direct Estimator #
    #########################################

    y_squared = y_samples_second.clone() ** 2 # SECOND

    model3 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=args.lr3)

    # Train the model3
    for epoch in range(args.num_epochs):

        outputs3 = model3(x_samples_second) # SECOND
        loss3 = criterion(outputs3.squeeze(), y_squared) # NOTE: USE y_squared
        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()
        
        '''
        if (epoch+1) % 100 == 0:
            print(f'Trial {i+1} - NN3 - Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss3.item():.4f}')
        '''

    with torch.no_grad():
      prediction3 = model3(x_samples_second).clone() # SECOND
      g_direct = prediction3 - prediction1 ** 2 # REUSE prediction1


    MSE_g_M2_holder[i] = torch.mean( (g_direct.squeeze() - g_values_second)**2 ) # SECOND


    ############################
    # Random Forest Regression #
    ############################

    x_samples_first = x_samples_first.clone().detach().cpu()
    #f_values_first = f_values_first.clone().detach().cpu()
    y_samples_first = y_samples_first.clone().detach().cpu()
    g_values_first = g_values_first.clone().detach().cpu()


    x_samples_second = x_samples_second.clone().detach().cpu()
    #f_values_second = f_values_second.clone().detach().cpu()
    y_samples_second = y_samples_second.clone().detach().cpu()
    g_values_second = g_values_second.clone().detach().cpu()


    RF_res_1 = RandomForestRegressor(max_depth=10, random_state=0)
    RF_res_2 = RandomForestRegressor(max_depth=10, random_state=0)

    RF_res_1.fit(x_samples_first, y_samples_first) # FIRST
    RF_prediction_1 = RF_res_1.predict(x_samples_second) # SECOND

    res_RF = y_samples_second - RF_prediction_1 # SECOND
    res_RF_squared = res_RF ** 2
    
    RF_res_2.fit(x_samples_second, res_RF_squared) # NOTE: USE res_RF_squared # SECOND
    prediction_RF_M3 = RF_res_2.predict(x_samples_second) # SECOND

    MSE_g_M3_holder[i] = np.mean( (prediction_RF_M3 - g_values_second.numpy())**2 )


    ###############################################
    # Random Forest Regression - Direct Estimator #
    ###############################################

    y_squared = y_samples_second ** 2 # SECOND

    RF_dir = RandomForestRegressor(max_depth=10, random_state=0)
    RF_dir.fit(x_samples_second, y_squared) # SECOND
    RF_dir_prediction = RF_dir.predict(x_samples_second) # SECOND

    prediction_RF_dir = RF_dir_prediction - RF_prediction_1 ** 2 # REUSE RF_prediction_1
    MSE_g_M4_holder[i] = np.mean( (prediction_RF_dir - g_values_second.numpy())**2 ) # SECOND

    if (i+1) % 10 == 0:
        print(f'[INFO] Trial {i+1} of {args.num_trials} complete,',\
        'mean of M1 MSE:', torch.mean(MSE_g_M1_holder[:(i+1)]))



########################
# Final Result (table) #
########################


print('\n')
print('mean of MSE_g_M1_holder:', torch.mean(MSE_g_M1_holder))
print('mean of MSE_g_M2_holder:', torch.mean(MSE_g_M2_holder))
print('mean of MSE_g_M3_holder:', np.mean(MSE_g_M3_holder))
print('mean of MSE_g_M4_holder:', np.mean(MSE_g_M4_holder))
print('\n')
print('std of MSE_g_M1_holder:', torch.std(MSE_g_M1_holder))
print('std of MSE_g_M2_holder:', torch.std(MSE_g_M2_holder))
print('std of MSE_g_M3_holder:', np.std(MSE_g_M3_holder))
print('std of MSE_g_M4_holder:', np.std(MSE_g_M4_holder))


torch.save(MSE_g_M1_holder, os.path.join(args.output_dir, f'{args.scenario}_split_M1_n{args.num_samples}_holder.pt') )
torch.save(MSE_g_M2_holder, os.path.join(args.output_dir, f'{args.scenario}_split_M2_n{args.num_samples}_holder.pt') )
np.save(os.path.join(args.output_dir, f'{args.scenario}_split_M3_n{args.num_samples}_holder.npy'), MSE_g_M3_holder)
np.save(os.path.join(args.output_dir, f'{args.scenario}_split_M4_n{args.num_samples}_holder.npy'), MSE_g_M4_holder)





















