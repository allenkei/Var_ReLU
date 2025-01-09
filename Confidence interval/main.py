import torch
from metrics.coverage import compute_coverage_probability

def main():
    """
    Main function to compute the average coverage probability.
    """
    # Configuration parameters
    num_trials = 100        # Number of Monte Carlo trials
    num_samples = 20000    # Number of samples per monte Carlo trial
    input_dim = 2          # Input dimension
    hidden_dim_f = 64      # Hidden layer size for f^ model
    hidden_dim_g = 64      # Hidden layer size for g^ model
    output_dim = 1         # Output dimension of models
    lr1 = 0.0001           # Learning rate for f^ model
    lr2 = 0.0001           # Learning rate for g^ model
    num_epochs = 1000      # Number of training epochs
    B = 1500               # Number of bootstrap iterations
    B_tilde = 1000         # Number of additional bootstrap iterations
    alpha = 0.05           # Significance level
    a_0 = alpha / (100 * torch.log(torch.tensor(num_samples)).item() ** 2)  # Global tuning parameter

    # Check for GPU or CPU usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Print start message
    print("[INFO] Starting coverage probability computation...")

    # Compute average coverage probability
    average_coverage, naive_coverage, interval_lengths = compute_coverage_probability(

        num_trials=num_trials,
        num_samples=num_samples,
        input_dim=input_dim,
        hidden_dim_f=hidden_dim_f,
        hidden_dim_g=hidden_dim_g,
        output_dim=output_dim,
        lr1=lr1,
        lr2=lr2,
        num_epochs=num_epochs,
        B=B,
        alpha=alpha,
        a_0=a_0,
        B_tilde=B_tilde,
        confidence_rep_quantity = 100,
        number_test_points=100,        
        device=device
    )

    # Print the results
    print(f"[RESULT] Average coverage probability: {average_coverage:.2%}")
    print(f"[RESULT] Naive coverage probability: {naive_coverage:.2%}")

if __name__ == "__main__":
    main()
