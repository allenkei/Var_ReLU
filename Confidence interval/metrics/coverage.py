"""
This script computes the average coverage probability for confidence intervals generated from Monte Carlo repetitions.

Key steps:
1. Generate synthetic data.
2. Partition data into training and testing sets.
3. Estimate model parameters for f(x) and g(x).
4. Compute residuals and bootstrap samples for confidence interval construction.
5. Evaluate the coverage probability of the confidence intervals.
"""



import torch
import numpy as np
from data_generation.data_processing import generate_data, partition_dataset
from data_generation.data_functions import f_function
from models.estimate_f import estimate_f_I1
from models.estimate_g import estimate_g_I2_with_An
from models.residuals import compute_normalized_residuals_and_empirical_distribution_with_An
from models.bootstrap import generate_bootstrapped_samples_and_train
from metrics.metrics import compute_a1_alpha, compute_a_and_b_alpha, construct_confidence_interval



def compute_coverage_probability(
    num_trials,
    num_samples,
    input_dim,
    hidden_dim_f,
    hidden_dim_g,
    output_dim,
    lr1,
    lr2,
    num_epochs,
    B,
    alpha,
    a_0,
    B_tilde,
    confidence_rep_quantity,
number_test_points,
    device
):
    """
    Compute the average coverage probability for the confidence intervals.

    Parameters:
    - num_trials (int): Number of outer Monte Carlo repetitions.
    - num_samples (int): Number of samples in each repetitions.
    - input_dim (int): Dimensionality of input data.
    - hidden_dim_f, hidden_dim_g (int): Hidden layer sizes for the neural networks.
    - output_dim (int): Output dimension of the neural networks.
    - lr1, lr2 (float): Learning rates for the neural networks.
    - num_epochs (int): Number of epochs for training.
    - B (int): Number of bootstrap iterations.
    - B_tilde (int): Number of bootstrap iterations used for averaging predictions.
    - alpha (float): Significance level.
    - a_0 (float): Tuning parameter.
    - confidence_rep_quantity (int): Number of inner Monte Carlo trials, used to compute the coverage in each outer monte carlo repetition.
    - number_test_points (int): Number of test points for evaluating coverage.
    - device (torch.device): PyTorch device to use for computation.

    Returns:
    - average_coverage (float): Average coverage probability across all outer monte carlo repetition.
    - naive_coverage (float): Naive coverage probability across all outer monte carlo repetition.
    """

    # Lists to store coverage rates for each outer monte carlo repetition
    outer_coverage_rates = []
    outer_naive_coverage_rates = []
    interval_lengths = {}

    for trial_out in range(num_trials):
        print(f"[INFO] Outer monte carlo repetition {trial_out + 1}/{num_trials}...")

        # Generate new data for this Outer monte carlo repetition
        new_data, An_values_new = generate_data(num_samples, confidence_rep_quantity, trial_out,input_dim)

         # Partition the data into subsets
        partitions_new = partition_dataset(new_data, num_samples)

        # Estimate f^(x) using I1
        f_models_new = estimate_f_I1(partitions_new, confidence_rep_quantity, input_dim, hidden_dim_f, output_dim, lr1, num_epochs, device)

        # Estimate g^(x) using I2 with A_n
        g_models_new = estimate_g_I2_with_An(partitions_new, f_models_new, confidence_rep_quantity, An_values_new, input_dim, hidden_dim_g, output_dim, lr2, num_epochs, device)

        # Compute normalized residuals and empirical distributions
        normalized_residuals_new, _ = compute_normalized_residuals_and_empirical_distribution_with_An(
            partitions_new, f_models_new, g_models_new, An_values_new, confidence_rep_quantity, device
        )

        # Generate bootstrap samples
        bootstrap_f_models_new, bootstrap_g_models_new = generate_bootstrapped_samples_and_train(
    partitions_new, f_models_new, g_models_new, An_values_new, normalized_residuals_new, confidence_rep_quantity, B+B_tilde,
    input_dim, hidden_dim_f, hidden_dim_g, output_dim, lr1, lr2, num_epochs, device
)

        # Compute a1(alpha) for each Outer monte carlo repetition
        a1_alpha_values = compute_a1_alpha(partitions_new,bootstrap_f_models_new,An_values_new,confidence_rep_quantity,B + B_tilde,alpha,device)

        # Compute a(alpha) and b(alpha) for each Outer monte carlo repetition
        a_alpha_values_new, b_alpha_values_new = compute_a_and_b_alpha(partitions_new, bootstrap_g_models_new, An_values_new, a1_alpha_values, confidence_rep_quantity, alpha, a_0,num_samples,B_tilde,device,num_samples)

        # Construct confidence intervals for each Outer monte carlo repetition
        confidence_intervals_new = construct_confidence_interval(a_alpha_values_new, b_alpha_values_new, confidence_rep_quantity,alpha,B_tilde)

        # Evaluate coverage for each test point
        coverage_individual = []
        coverage_individual_naive = []
        
        for trial_in in range(number_test_points):
            print(f"[INFO] test point number {trial_in + 1}/{number_test_points}...")
            torch.manual_seed(trial_in+100*num_trials+(trial_out)*confidence_rep_quantity)
            np.random.seed(trial_in+100*num_trials+(trial_out)*confidence_rep_quantity)

            # Generate a new test point
            x_new = torch.rand(1, input_dim).to(device)
            f_true_values = f_function(x_new)
            trial_coverage_count = 0
            trial_naive_coverage_count = 0
            # Store interval lengths for this test point
            interval_lengths[f"Test Point {trial_in + 1}"] = {}
            for trial_in_1 in range(confidence_rep_quantity):
                # Extract corresponding confidence interval
                lower_bound, upper_bound = confidence_intervals_new[trial_in_1]

                # Average predictions over bootstrap models
                accumulated_predictions = torch.zeros_like(x_new[:, 0]).to(device)
                for i in range(B_tilde, len(bootstrap_f_models_new[trial_in_1])):
                    prediction = bootstrap_f_models_new[trial_in_1][i](x_new).clone().squeeze()
                    accumulated_predictions += prediction.clamp(-An_values_new[trial_in_1], An_values_new[trial_in_1])
                f_predictions_B_plus_1 = accumulated_predictions / (len(bootstrap_f_models_new[trial_in_1]) - B_tilde)

                # Adjust confidence interval
                lower_bound += f_predictions_B_plus_1
                upper_bound += f_predictions_B_plus_1

                # Check if true value is within the interval
                if lower_bound <= f_true_values <= upper_bound:
                    trial_coverage_count += 1

                # Print the confidence interval and true value for this trial
                print(f"[DEBUG] In test point number {trial_in + 1}: The Interval for dataset number {trial_in_1 + 1} is = [{lower_bound.item():.4f}, {upper_bound.item():.4f}], "
                  f"True value f(test point) = {f_true_values.item():.4f}")
                # Compute interval length
                interval_length = (upper_bound - lower_bound).item()
                interval_lengths[f"Test Point {trial_in + 1}"][f"Dataset {trial_in_1 + 1}"] = interval_length
                print(f"[DEBUG] The length of the Interval for dataset number {trial_in_1 + 1} is = {interval_length:.4f}")
                # Compute naive confidence interval
                bootstrap_predictions = []
                for i in range(len(bootstrap_f_models_new[trial_in_1])):
                    with torch.no_grad():
                        prediction = bootstrap_f_models_new[trial_in_1][i](x_new).clone().squeeze()
                        bootstrap_predictions.append(prediction.item())
                bootstrap_predictions.sort()
                naive_lower = np.percentile(bootstrap_predictions, 100 * alpha / 2)
                naive_upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2))
                # Print debug information
                print(f"[DEBUG] In test point number {trial_in + 1}: The Naive Interval for dataset number {trial_in_1 + 1} is = [{naive_lower:.4f}, {naive_upper:.4f}], "
                  f"True value f(test point) = {f_true_values.item():.4f}")
                naive_length = naive_upper - naive_lower
                interval_lengths[f"Test Point {trial_in + 1}"][f"Dataset {trial_in_1 + 1} (Naive)"] = naive_length
                print(f"[DEBUG] The length of the Naive Interval for dataset number {trial_in_1 + 1} is = {naive_length:.4f}")
                # Check naive coverage
                if naive_lower <= f_true_values <= naive_upper:
                    trial_naive_coverage_count += 1
            
            coverage_for_new_data=trial_coverage_count / confidence_rep_quantity
            coverage_individual.append(coverage_for_new_data)
            coverage_for_new_data_naive=trial_naive_coverage_count / confidence_rep_quantity
            coverage_individual_naive.append(coverage_for_new_data_naive)


        coverage_individual_rate=sum(coverage_individual) / number_test_points #number of test points
        coverage_individual_rate_naive=sum(coverage_individual_naive) / number_test_points #number of test points
        

        # Store coverage rates
        outer_coverage_rates.append(coverage_individual_rate)
        outer_naive_coverage_rates.append(coverage_individual_rate_naive)

    # Compute average coverage probabilities
    average_coverage = sum(outer_coverage_rates) / num_trials
    naive_coverage = sum(outer_naive_coverage_rates) / num_trials
    return average_coverage, naive_coverage, interval_lengths
