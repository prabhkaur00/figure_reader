import pandas as pd
import matplotlib.pyplot as plt

def plot_three_csvs(est1_path, est2_path, gt_path, output_filename):
    # Load data with automatic delimiter detection
    # sep=None with engine='python' handles both comma and tab separated files
    df_est1 = pd.read_csv(est1_path, sep=None, engine='python')
    df_est2 = pd.read_csv(est2_path, sep=None, engine='python')
    df_gt = pd.read_csv(gt_path, sep=None, engine='python')

    # Create plot
    plt.figure(figsize=(12, 7))

    # Plot Ground Truth
    plt.plot(df_gt.iloc[:, 0], df_gt.iloc[:, 1], 
             label="Ground Truth", color='black', linewidth=2, zorder=3)

    # Plot Estimate 1
    plt.plot(df_est1.iloc[:, 0], df_est1.iloc[:, 1], 
             label="Matplotlib Estimate", alpha=0.7, linestyle='--')

    # Plot Estimate 2
    plt.plot(df_est2.iloc[:, 0], df_est2.iloc[:, 1], 
             label="OpenCV Estimate", alpha=0.7, linestyle=':')

    # Formatting
    plt.title("Comparison of Raman Spectra: Estimates vs. Ground Truth")
    plt.xlabel(df_gt.columns[0])
    plt.ylabel(df_gt.columns[1])
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_three_csvs(
        "./OpenCV/raman1_matplotlib.csv", 
        "./OpenCV/raman1_opencv.csv", 
        "./OpenCV/raman1_gt.csv", 
        "./Comparison/raman1_comparison.png"
    )