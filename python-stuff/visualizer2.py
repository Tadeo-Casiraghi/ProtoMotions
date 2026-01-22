import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "multiple_arrays.npz"   # Updated file name
DT = 1.0 / 30.0                     # Physics step size
# ---------------------

def plot_data():
    try:
        # 1. Load the data
        print(f"Loading {FILE_PATH}...")
        # np.savez creates a dictionary-like object
        loaded_data = np.load(FILE_PATH)
        
        # Extract individual arrays using the keys you defined in np.savez
        prismatic = loaded_data['prismatic'].flatten()
        rot_x = loaded_data['rotx'].flatten()*180/np.pi
        rot_y = loaded_data['roty'].flatten()*180/np.pi
        rot_z = loaded_data['rotz'].flatten()*180/np.pi
        
        num_frames = len(prismatic)
        print(f"Loaded {num_frames} frames.")

        # 2. Create Time Axis
        time_axis = np.linspace(0, num_frames * DT, num_frames)

        # 3. Plotting
        # We use subplots: 2 Rows, 1 Column. Share X axis so zooming one zooms both.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # --- Plot 1: Prismatic Position ---
        ax1.plot(time_axis, prismatic, label='Prismatic Ext.', color='black', linewidth=1.5)
        ax1.set_title("Prismatic Joint Position", fontsize=14)
        ax1.set_ylabel("Position (meters)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')
        ax1.set_ylim((-0.06, 0.01))

        # --- Plot 2: Rotations (XYZ) ---
        ax2.plot(time_axis, rot_x, label='Rot X', color='red', linewidth=1.0, alpha=0.8)
        ax2.plot(time_axis, rot_y, label='Rot Y', color='green', linewidth=1.0, alpha=0.8)
        ax2.plot(time_axis, rot_z, label='Rot Z', color='blue', linewidth=1.0, alpha=0.8)
        
        ax2.set_title("Joint Rotations (XYZ)", fontsize=14)
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Rotation (rads/degs)", fontsize=12) # Update based on your units
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        plt.tight_layout() # Fixes spacing between plots
        
        # Show plot
        print("Displaying plot...")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'.")
    except KeyError as e:
        print(f"Error: The key {e} was not found in the .npz file. Check your save function.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_data()