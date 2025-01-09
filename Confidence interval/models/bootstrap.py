import torch
import torch.nn as nn
import torch.optim as optim
from models.nn_model import NN

def generate_bootstrapped_samples_and_train(
    partitions, f_models, g_models, An_values, normalized_residuals, confidence_rep_quantity, B,
    input_dim, hidden_dim_f, hidden_dim_g, output_dim, lr1, lr2, num_epochs, device
):
    """
    Generate bootstrap samples and train f^ and g^ models for each trial.

    Parameters:
    tiral=specific dataset generated among all needed to compute coverage in a fixed test point, in a monte carlo repetition
    - partitions (dict): Dataset partitions (I1, I2, I3, I4) for each trial.
    - f_models (list): List of trained f^ models for each trial.
    - g_models (list): List of trained g^ models for each trial.
    - An_values (list): List of A_n values for each trial.
    - normalized_residuals (list): Normalized residuals for each trial.
    - confidence_rep_quantity (int): Number of trials for coverage evaluation.
    - B (int): Number of bootstrap iterations.
    - input_dim (int): Input dimension of the dataset.
    - hidden_dim_f (int): Hidden layer size for the f^ model.
    - hidden_dim_g (int): Hidden layer size for the g^ model.
    - output_dim (int): Output dimension of the models.
    - lr1, lr2 (float): Learning rates for the f^ and g^ models.
    - num_epochs (int): Number of training epochs for each model.
    - device (torch.device): Device to run the computations on.

    Returns:
    - bootstrap_f_models (list): List of lists of trained f^ models for each bootstrap iteration, for all trials.
    - bootstrap_g_models (list): List of lists of trained g^ models for each bootstrap iteration, for all trials.
    """
    bootstrap_f_models = []  # Store f^ models for each trial
    bootstrap_g_models = []  # Store g^ models for each trial

    for trial in range(confidence_rep_quantity):
        print(f"[INFO] Generating bootstrap samples for dataset {trial + 1}/{confidence_rep_quantity} corresponding to a fixed dataset used to compute coverage, in a fixed monte carlo repetition...")
        
        I3 = partitions['I3'][trial]
        x_samples_I3 = I3[:, 3:].to(device)
        y_samples_I3 = I3[:, 2].to(device)
        f_values = I3[:, 0].to(device)
        g_values = I3[:, 1].to(device)
        An = An_values[trial]
        model_f = f_models[trial]
        model_g = g_models[trial]
        
        normalized_residuals_trial = normalized_residuals[trial]

        trial_f_models = []  # List for this trial's f^ models
        trial_g_models = []  # List for this trial's g^ models
        for b in range(B + 1):
            print(f"[INFO] Bootstrap {b+1}/{B +1}...")
            with torch.no_grad():
                bootstrap_residuals = normalized_residuals_trial[
                    torch.randint(0, len(normalized_residuals_trial), (len(normalized_residuals_trial),))
                ]
                g_predictions = model_g(x_samples_I3).clone().squeeze().clamp(-An, An)
                f_predictions_clamped = model_f(x_samples_I3).clone().squeeze().clamp(-An, An)
                bootstrap_y = f_predictions_clamped + bootstrap_residuals * torch.sqrt(torch.abs(g_predictions))

            # Train f^ model on bootstrap samples
            model_f_bootstrap = NN(input_dim, hidden_dim_f, output_dim).to(device)
            optimizer_f = optim.Adam(model_f_bootstrap.parameters(), lr=lr1)
            criterion = nn.MSELoss()

            for epoch in range(num_epochs):
                outputs = model_f_bootstrap(x_samples_I3)
                loss = criterion(outputs.squeeze(), bootstrap_y)
                optimizer_f.zero_grad()
                loss.backward()
                optimizer_f.step()

            # Train g^ model on bootstrap samples
            model_g_bootstrap = NN(input_dim, hidden_dim_g, output_dim).to(device)
            optimizer_g = optim.Adam(model_g_bootstrap.parameters(), lr=lr2)

            with torch.no_grad():
                residuals_squared_bootstrap = (bootstrap_y - model_f_bootstrap(x_samples_I3).clone().squeeze()) ** 2

            for epoch in range(num_epochs):
                outputs = model_g_bootstrap(x_samples_I3)
                loss = criterion(outputs.squeeze(), residuals_squared_bootstrap)
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

            # Append the models for this bootstrap iteration
            trial_f_models.append(model_f_bootstrap)
            trial_g_models.append(model_g_bootstrap)
        
        # Append the trial-specific lists to the overall lists
        bootstrap_f_models.append(trial_f_models)
        bootstrap_g_models.append(trial_g_models)
        

    return bootstrap_f_models, bootstrap_g_models