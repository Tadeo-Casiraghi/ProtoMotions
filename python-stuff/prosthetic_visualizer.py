import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "suspension_slide.npy"  # Name of your file
DT = 1.0 / 30.0                     # Physics step size (change to 1/50 or 1/100 if needed)
# ---------------------

def plot_data():
    try:
        # 1. Load the data
        print(f"Loading {FILE_PATH}...")
        data = np.load(FILE_PATH)
        
        # If the data saved as a list of single-element tensors/arrays, flatten it
        data = data.flatten()
        
        num_frames = len(data)
        print(f"Loaded {num_frames} frames.")

        # 2. Create Time Axis
        # This converts "Step 1, Step 2..." into "0.0s, 0.016s..."
        time_axis = np.linspace(0, num_frames * DT, num_frames)

        # 3. Plotting
        plt.figure(figsize=(10, 6))
        
        plt.plot(time_axis, data, label='Prismatic Joint Position', color='blue', linewidth=1.5)
        
        # Styling
        plt.title(f"Prosthetic Joint Movement ({num_frames} steps)", fontsize=16)
        plt.xlabel("Time (seconds)", fontsize=14)
        plt.ylabel("Position (meters)", fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Show plot
        print("Displaying plot...")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'. Make sure you are in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_data()