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

  parser.add_argument('--n', default=5000, type=int)
  parser.add_argument('--data', default='housing', type=str)
  parser.add_argument('--input_dim', default=2, type=int)
  parser.add_argument('--hidden_dim', default=64, type=int)
  parser.add_argument('--output_dim', default=1) 
  parser.add_argument('--lr1', default=0.001) # M1 estimator based on residuals
  parser.add_argument('--lr2', default=0.001) # M1 estimator based on residuals
  parser.add_argument('--lr3', default=0.001) # M2 direct estimator
  parser.add_argument('--num_epochs', default=1000)
  parser.add_argument('--num_trials', default=100, type=int)
  parser.add_argument('--data_dir', default='./data_real/')
  parser.add_argument('--output_dir', default='./result_real/')
  parser.add_argument('-f', required=False) # needed in Colab 

  return parser.parse_args()


args = parse_args(); print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)
os.makedirs(args.output_dir, exist_ok=True)
df_all_data = pd.read_csv(os.path.join(args.data_dir,f"{args.data}.csv"))


########
# DATA #
########

df_all_data['avg_occupancy'] = df_all_data['population'] / df_all_data['households']
df_all_data['log_median_house_value'] = np.log(df_all_data['median_house_value'])
df_selected = df_all_data[['avg_occupancy', 'median_income', 'log_median_house_value']]


lower_avg_occupancy = df_selected['avg_occupancy'].quantile(0.025)
upper_avg_occupancy = df_selected['avg_occupancy'].quantile(0.975)
lower_median_income = df_selected['median_income'].quantile(0.025)
upper_median_income = df_selected['median_income'].quantile(0.975)


df_selected = df_selected[
    (df_selected['avg_occupancy'] >= lower_avg_occupancy) & (df_selected['avg_occupancy'] <= upper_avg_occupancy) &
    (df_selected['median_income'] >= lower_median_income) & (df_selected['median_income'] <= upper_median_income)
]

print(f'[INFO] {args.data} data size:',df_selected.shape)



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



Prop_g_M1_95_holder = torch.zeros(args.num_trials) # estimator based on residuals
Prop_g_M1_90_holder = torch.zeros(args.num_trials) # estimator based on residuals
Prop_g_M2_95_holder = torch.zeros(args.num_trials) # direct estimator
Prop_g_M2_90_holder = torch.zeros(args.num_trials) # direct estimator
Prop_g_M3_95_holder = np.zeros(args.num_trials) # RF - estimator based on residuals
Prop_g_M3_90_holder = np.zeros(args.num_trials) # RF - estimator based on residuals
Prop_g_M4_95_holder = np.zeros(args.num_trials) # RF - direct estimator
Prop_g_M4_90_holder = np.zeros(args.num_trials) # RF - direct estimator
criterion = nn.MSELoss()



for i in range(args.num_trials):

  df_selected = df_selected.sample(frac=1).reset_index(drop=True)
  df_train = df_selected.iloc[:args.n,:]
  df_test = df_selected.iloc[args.n:,:]

  x_train_sample = df_train[['avg_occupancy', 'median_income']]
  y_train_sample = df_train['log_median_house_value']
  x_test_sample = df_test[['avg_occupancy', 'median_income']]
  y_test_sample = df_test['log_median_house_value']

  x_train_sample = torch.tensor(x_train_sample.values, dtype=torch.float32).to(device)
  y_train_sample = torch.tensor(y_train_sample.values, dtype=torch.float32).to(device)
  x_test_sample = torch.tensor(x_test_sample.values, dtype=torch.float32).to(device)
  y_test_sample = torch.tensor(y_test_sample.values, dtype=torch.float32).to(device)

  ####################
  # Neural Network 1 #
  ####################

  model1 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
  optimizer1 = optim.Adam(model1.parameters(), lr=args.lr1)


  # Training the model1
  for epoch in range(args.num_epochs):

      outputs1 = model1(x_train_sample) 
      loss1 = criterion(outputs1.squeeze(), y_train_sample)
      optimizer1.zero_grad()
      loss1.backward()
      optimizer1.step()
      

  #####################
  # Residuals squared #
  #####################

  with torch.no_grad():
    prediction1 = model1(x_train_sample).clone() 
    res1 = y_train_sample - prediction1.squeeze() 
    res1_squared = res1 ** 2

  ####################
  # Neural Network 2 #
  ####################

  model2 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
  optimizer2 = optim.Adam(model2.parameters(), lr=args.lr2)

  # Train the model2
  for epoch in range(args.num_epochs):

      outputs2 = model2(x_train_sample) 
      loss2 = criterion(outputs2.squeeze(), res1_squared) # NOTE: USE residual squared 
      optimizer2.zero_grad()
      loss2.backward()
      optimizer2.step()
  
  # Test set
  with torch.no_grad():
    test_prediction1 = model1(x_test_sample).clone() 
    test_prediction2 = model2(x_test_sample).clone() 
    test_prediction2 = torch.where(test_prediction2 < 0, torch.tensor(0.0), test_prediction2)
  
  # 95% interval
  lower_bound = test_prediction1 - 1.96 * torch.sqrt(test_prediction2)
  upper_bound = test_prediction1 + 1.96 * torch.sqrt(test_prediction2)

  is_in_interval = (y_test_sample >= lower_bound.squeeze()) & (y_test_sample <= upper_bound.squeeze())
  Prop_g_M1_95_holder[i] = is_in_interval.float().mean().item()

  # 90% interval
  lower_bound = test_prediction1 - 1.65 * torch.sqrt(test_prediction2)
  upper_bound = test_prediction1 + 1.65 * torch.sqrt(test_prediction2)

  is_in_interval = (y_test_sample >= lower_bound.squeeze()) & (y_test_sample <= upper_bound.squeeze())
  Prop_g_M1_90_holder[i] = is_in_interval.float().mean().item()


  #########################################
  # Neural Network 3 for Direct Estimator #
  #########################################

  y_squared = y_train_sample.clone() ** 2

  model3 = NN(args.input_dim, args.hidden_dim, args.output_dim).to(device)
  optimizer3 = optim.Adam(model3.parameters(), lr=args.lr3)

  # Train the model3
  for epoch in range(args.num_epochs):

      outputs3 = model3(x_train_sample)
      loss3 = criterion(outputs3.squeeze(), y_squared) # NOTE: USE y_squared
      optimizer3.zero_grad()
      loss3.backward()
      optimizer3.step()

  # Test set
  with torch.no_grad():
    test_prediction3 = model3(x_test_sample).clone()
    test_g_direct = test_prediction3 - test_prediction1 ** 2 # REUSE test_prediction1
    test_g_direct = torch.where(test_g_direct < 0, torch.tensor(0.0), test_g_direct)
  
    # 95% interval
    lower_bound = test_prediction1 - 1.96 * torch.sqrt(test_g_direct)
    upper_bound = test_prediction1 + 1.96 * torch.sqrt(test_g_direct)

    is_in_interval = (y_test_sample >= lower_bound.squeeze()) & (y_test_sample <= upper_bound.squeeze())
    Prop_g_M2_95_holder[i] = is_in_interval.float().mean().item()
    
    # 90% interval
    lower_bound = test_prediction1 - 1.65 * torch.sqrt(test_g_direct)
    upper_bound = test_prediction1 + 1.65 * torch.sqrt(test_g_direct)

    is_in_interval = (y_test_sample >= lower_bound.squeeze()) & (y_test_sample <= upper_bound.squeeze())
    Prop_g_M2_90_holder[i] = is_in_interval.float().mean().item()


  ############################
  # Random Forest Regression #
  ############################

  x_train_sample = x_train_sample.clone().detach().cpu()
  y_train_sample = y_train_sample.clone().detach().cpu()

  x_test_sample = x_test_sample.clone().detach().cpu()
  y_test_sample = y_test_sample.clone().detach().cpu()


  RF_res_1 = RandomForestRegressor(max_depth=10, random_state=0)
  RF_res_2 = RandomForestRegressor(max_depth=10, random_state=0)

  RF_res_1.fit(x_train_sample, y_train_sample)
  RF_prediction_1 = RF_res_1.predict(x_train_sample)

  res_RF = y_train_sample - RF_prediction_1 
  res_RF_squared = res_RF ** 2
  
  RF_res_2.fit(x_train_sample, res_RF_squared) # NOTE: USE res_RF_squared
  
  # Testing set
  test_RF_prediction1 = RF_res_1.predict(x_test_sample) # f()
  test_RF_prediction2 = RF_res_2.predict(x_test_sample) # g()
  test_RF_prediction2[test_RF_prediction2 < 0] = 0.0

  # 95% interval
  lower_bound = test_RF_prediction1 - 1.96 * np.sqrt(test_RF_prediction2)
  upper_bound = test_RF_prediction1 + 1.96 * np.sqrt(test_RF_prediction2)

  is_in_interval = (y_test_sample.cpu().numpy() >= lower_bound.squeeze()) & (y_test_sample.cpu().numpy() <= upper_bound.squeeze())
  Prop_g_M3_95_holder[i] = is_in_interval.mean().item()
  
  # 90% interval
  lower_bound = test_RF_prediction1 - 1.65 * np.sqrt(test_RF_prediction2)
  upper_bound = test_RF_prediction1 + 1.65 * np.sqrt(test_RF_prediction2)

  is_in_interval = (y_test_sample.cpu().numpy() >= lower_bound.squeeze()) & (y_test_sample.cpu().numpy() <= upper_bound.squeeze())
  Prop_g_M3_90_holder[i] = is_in_interval.mean().item()


  ###############################################
  # Random Forest Regression - Direct Estimator #
  ###############################################

  y_squared = y_train_sample ** 2

  RF_dir = RandomForestRegressor(max_depth=10, random_state=0)
  RF_dir.fit(x_train_sample, y_squared)
  
  # Testing set
  test_RF_dir_prediction = RF_dir.predict(x_test_sample)
  test_g_RF_dir = test_RF_dir_prediction - test_RF_prediction1 ** 2 # REUSE test_RF_prediction1
  test_g_RF_dir[test_g_RF_dir < 0] = 0.0

  lower_bound = test_RF_prediction1 - 1.96 * np.sqrt(test_g_RF_dir)
  upper_bound = test_RF_prediction1 + 1.96 * np.sqrt(test_g_RF_dir)

  is_in_interval = (y_test_sample.cpu().numpy() >= lower_bound.squeeze()) & (y_test_sample.cpu().numpy() <= upper_bound.squeeze())
  Prop_g_M4_95_holder[i] = is_in_interval.mean().item()

  lower_bound = test_RF_prediction1 - 1.65 * np.sqrt(test_g_RF_dir)
  upper_bound = test_RF_prediction1 + 1.65 * np.sqrt(test_g_RF_dir)

  is_in_interval = (y_test_sample.cpu().numpy() >= lower_bound.squeeze()) & (y_test_sample.cpu().numpy() <= upper_bound.squeeze())
  Prop_g_M4_90_holder[i] = is_in_interval.mean().item()


  
  if (i+1) % 10 == 0:
      print(f'[INFO] Trial {i+1} of {args.num_trials} complete,',\
      'mean of M1 95% Prop:', torch.mean(Prop_g_M1_95_holder[:(i+1)]),
      'mean of M1 90% Prop:', torch.mean(Prop_g_M1_90_holder[:(i+1)]))




print('mean of Prop_g_M1_95_holder:', torch.mean(Prop_g_M1_95_holder))
print('mean of Prop_g_M1_90_holder:', torch.mean(Prop_g_M1_90_holder))
print('mean of Prop_g_M2_95_holder:', torch.mean(Prop_g_M2_95_holder))
print('mean of Prop_g_M2_90_holder:', torch.mean(Prop_g_M2_90_holder))
print('mean of Prop_g_M3_95_holder:', np.mean(Prop_g_M3_95_holder))
print('mean of Prop_g_M3_90_holder:', np.mean(Prop_g_M3_90_holder))
print('mean of Prop_g_M4_95_holder:', np.mean(Prop_g_M4_95_holder))
print('mean of Prop_g_M4_90_holder:', np.mean(Prop_g_M4_90_holder))

print('std of Prop_g_M1_95_holder:', torch.std(Prop_g_M1_95_holder))
print('std of Prop_g_M1_90_holder:', torch.std(Prop_g_M1_90_holder))
print('std of Prop_g_M2_95_holder:', torch.std(Prop_g_M2_95_holder))
print('std of Prop_g_M2_90_holder:', torch.std(Prop_g_M2_90_holder))
print('std of Prop_g_M3_95_holder:', np.std(Prop_g_M3_95_holder))
print('std of Prop_g_M3_90_holder:', np.std(Prop_g_M3_90_holder))
print('std of Prop_g_M4_95_holder:', np.std(Prop_g_M4_95_holder))
print('std of Prop_g_M4_90_holder:', np.std(Prop_g_M4_90_holder))

torch.save(Prop_g_M1_95_holder, os.path.join(args.output_dir, f'{args.data}_n{args.n}_M1_95.pt') )
torch.save(Prop_g_M1_90_holder, os.path.join(args.output_dir, f'{args.data}_n{args.n}_M1_90.pt') )
torch.save(Prop_g_M2_95_holder, os.path.join(args.output_dir, f'{args.data}_n{args.n}_M2_95.pt') )
torch.save(Prop_g_M2_90_holder, os.path.join(args.output_dir, f'{args.data}_n{args.n}_M2_90.pt') )
np.save(os.path.join(args.output_dir, f'{args.data}_n{args.n}_M3_95.npy'), Prop_g_M3_95_holder)
np.save(os.path.join(args.output_dir, f'{args.data}_n{args.n}_M3_90.npy'), Prop_g_M3_90_holder)
np.save(os.path.join(args.output_dir, f'{args.data}_n{args.n}_M4_95.npy'), Prop_g_M4_95_holder)
np.save(os.path.join(args.output_dir, f'{args.data}_n{args.n}_M4_90.npy'), Prop_g_M4_90_holder)






















