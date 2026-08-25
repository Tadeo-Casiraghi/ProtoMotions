import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Set up argument parsing to easily choose test parameters from the terminal
    parser = argparse.ArgumentParser(description="Compare test movement data against the Ground Truth.")
    parser.add_argument("--dir", type=str, default="data2", help="Directory where files are stored")
    
    args = parser.parse_args()

    tests = [(360,16,4)]
    
    # Define file paths
    ground_truth_path = os.path.join(args.dir, "movement_freq4800hz_pos32_vel4.csv")
    test_file_paths = [os.path.join(args.dir, f"movement_freq{freq}hz_pos{pos}_vel{vel}.csv") for freq, pos, vel in tests]
    simplified_file_path = os.path.join(f"movement_simplified.csv")

    # Check if all files exist before opening
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at '{ground_truth_path}'")
        return
    if not os.path.exists(simplified_file_path):
        print(f"No simplified reference at '{simplified_file_path}'")
        simplified_exists = False
    else:
        simplified_exists = True
    for test_file_path in test_file_paths:
        if not os.path.exists(test_file_path):
            print(f"Error: Test file not found at '{test_file_path}'")
            return
        
    # Load data
    print(f"Loading ground truth: {ground_truth_path}")
    df_gt = pd.read_csv(ground_truth_path)
    
    if simplified_exists:
        print(f"Loading simplified truth: {simplified_file_path}")
        df_simplified= pd.read_csv(simplified_file_path)
        
    
    
    # Identify target columns to plot (exclude Time and Mean_Wall_Time_Sec)
    exclude_cols = {'Time', 'Mean_Wall_Time_Sec'}
    columns_to_plot = [col for col in df_gt.columns if col not in exclude_cols]
   

    if not columns_to_plot:
        print("No valid data columns found to plot.")
        return
        
    num_plots = len(columns_to_plot)
    
    # Create aligned subplots for each feature sharing the same X-axis (Time)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)
    
    # Handle the case where there's only 1 column to plot to keep axes iterable
    if num_plots == 1:
        axes = [axes]
        
    for ax, col in zip(axes, columns_to_plot):
        if col in df_gt.columns:
            # Plot Ground Truth - Dotted line
            if col != "Pistoning (suspension_slide)":
                ax.plot(df_gt['Time'], df_gt[col]*180/3.1415, label='Ground Truth (4800hz, pos32, vel16)', 
                        linestyle=':', color='black', linewidth=1.5)
            else:
                ax.plot(df_gt['Time'], df_gt[col], label='Ground Truth (4800hz, pos32, vel16)', 
                        linestyle=':', color='black', linewidth=1.5)
                
            if simplified_exists:
                if col != "Pistoning (suspension_slide)":
                    ax.plot(df_simplified['Time'], df_simplified[col]*180/3.1415, label='simplified', 
                            linestyle=':', color='red', linewidth=1.5)
                else:
                     ax.plot(df_simplified['Time'], df_simplified[col], label='simplified', 
                            linestyle=':', color='red', linewidth=1.5)
            
            for test_file_path, name in zip(test_file_paths, tests):
                print(f"Loading test file: {test_file_path}")
                df_test = pd.read_csv(test_file_path)
                # Plot Test Data - Solid line
                if col != "Pistoning (suspension_slide)":
                    ax.plot(df_test['Time'], df_test[col]*180/3.1415, label=f'Test {name}', 
                            linestyle='-', linewidth=1.2, alpha=0.8)
                else:
                    ax.plot(df_test['Time'], df_test[col], label=f'Test {name}', 
                            linestyle='-', linewidth=1.2, alpha=0.8)
            
            ax.set_ylabel(col)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right')
        else:
            print(f"Warning: Column '{col}' not found in the test file.")
            
    axes[-1].set_xlabel('Time')
    plt.suptitle(f"Comparison: Test vs Ground Truth", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    # Save the resulting visualization
    output_image = f"comparison_genera.png"
    print(f"Plot saved successfully as '{output_image}'.")
    plt.savefig(output_image, dpi=300)

if __name__ == "__main__":
    main()