import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "multiple_arrays.npz" # Ensure path is correct
DT = 1.0 / 30.0
skin_names = [
    "skin_box_posterior_top", "skin_box_medial_top",
    "skin_box_anterior_top", "skin_box_lateral_top",
    "skin_box_posterior_bottom", "skin_box_medial_bottom",
    "skin_box_anterior_bottom", "skin_box_lateral_bottom"
]
# ---------------------

def plot_data():
    try:
        print(f"Loading {FILE_PATH}...")
        loaded_data = np.load(FILE_PATH)
        
        # --- Load Kinematics ---
        prismatic = loaded_data['prismatic'].flatten()
        rot_x = loaded_data['rotx'].flatten() * 180 / np.pi
        rot_y = loaded_data['roty'].flatten() * 180 / np.pi
        rot_z = loaded_data['rotz'].flatten() * 180 / np.pi
        
        # --- Load Forces ---
        # 1. Local (Skin) Frame
        if 'skin_forces' in loaded_data:
            local_forces = loaded_data['skin_forces']
        else:
            print("Warning: 'skin_forces' (Local) not found.")
            local_forces = None

        # 2. Knee Frame (We need this for Fig 3 AND Fig 4 now)
        if 'skin_forces_knee' in loaded_data:
            knee_forces = loaded_data['skin_forces_knee']
        else:
            print("Warning: 'skin_forces_knee' not found.")
            knee_forces = None

        # Time Axis
        num_frames = len(prismatic)
        time_axis = np.linspace(0, num_frames * DT, num_frames)

        # =========================================================
        # FIGURE 1: KINEMATICS
        # =========================================================
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig1.canvas.manager.set_window_title('Fig 1: Kinematics')

        ax1.plot(time_axis, prismatic, 'k', linewidth=1.5, label='Prismatic Ext.')
        ax1.set_title("Prismatic Joint Position", fontsize=14)
        ax1.set_ylabel("Position (m)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        ax2.plot(time_axis, rot_x, 'r', label='Rot X')
        ax2.plot(time_axis, rot_y, 'g', label='Rot Y')
        ax2.plot(time_axis, rot_z, 'b', label='Rot Z')
        ax2.set_title("Joint Rotations (XYZ)", fontsize=14)
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Angle (deg)", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        plt.tight_layout()

        # =========================================================
        # FIGURE 2: LOCAL FRAME (Normal vs Shear) - Per Box
        # =========================================================
        if local_forces is not None:
            fig2, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
            fig2.canvas.manager.set_window_title('Fig 2: Local Forces (Normal vs Shear)')
            fig2.suptitle("Local Frame: Normal (Z) vs Shear (XY Mag)", fontsize=16)
            axes_flat = axes.flatten()

            for i, name in enumerate(skin_names):
                if i >= local_forces.shape[1]: break
                ax = axes_flat[i]
                
                # Z is Normal (Compression)
                f_normal = local_forces[:, i, 2]
                # Magnitude of X and Y is Shear
                f_shear = np.sqrt(local_forces[:, i, 0]**2 + local_forces[:, i, 1]**2)

                ax.plot(time_axis, f_normal, color='blue', linewidth=1.2, label='Normal (Z)')
                ax.plot(time_axis, f_shear, color='red', linewidth=1.2, alpha=0.8, label='Shear (XY)')
                ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

                ax.set_title(name, fontsize=10, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.5)
                if i == 0: ax.legend(loc='upper right', fontsize='small')

            for ax in axes[-1, :]: ax.set_xlabel("Time (s)")
            for ax in axes[:, 0]: ax.set_ylabel("Force (N)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # =========================================================
        # FIGURE 3: KNEE FRAME (Bone Relative) - Per Box
        # =========================================================
        if knee_forces is not None:
            fig3, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
            fig3.canvas.manager.set_window_title('Fig 3: Knee Frame Forces (Per Box)')
            fig3.suptitle("Individual Sensor Forces (Knee Frame)", fontsize=16)
            axes_flat = axes.flatten()

            for i, name in enumerate(skin_names):
                if i >= knee_forces.shape[1]: break
                ax = axes_flat[i]
                
                fx = knee_forces[:, i, 0]
                fy = knee_forces[:, i, 1]
                fz = knee_forces[:, i, 2]

                ax.plot(time_axis, fx, 'r', alpha=0.6, label='Knee X')
                ax.plot(time_axis, fy, 'g', alpha=0.6, label='Knee Y')
                ax.plot(time_axis, fz, 'b', alpha=0.6, label='Knee Z')

                ax.set_title(name, fontsize=10, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.5)
                if i == 0: ax.legend(loc='upper right', fontsize='small')

            for ax in axes[-1, :]: ax.set_xlabel("Time (s)")
            for ax in axes[:, 0]: ax.set_ylabel("Force (N)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # =========================================================
        # FIGURE 4: KNEE FRAME (Total Net Force)
        # =========================================================
        if knee_forces is not None:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            fig4.canvas.manager.set_window_title('Fig 4: Net Knee Frame Force')

            # 1. Sum vectors first (Vector Sum of all sensors in Knee Frame)
            # knee_forces shape is [Time, Sensors, 3] -> Sum over axis 1
            net_vector = np.sum(knee_forces, axis=1) 
            
            k_fx = net_vector[:, 0]
            k_fy = net_vector[:, 1]
            k_fz = net_vector[:, 2]
            
            # 2. Magnitude of the result
            net_mag = np.linalg.norm(net_vector, axis=1)

            ax4.plot(time_axis, k_fx, 'r', alpha=0.6, label='Knee X')
            ax4.plot(time_axis, k_fy, 'g', alpha=0.6, label='Knee Y')
            ax4.plot(time_axis, k_fz, 'b', alpha=0.6, label='Knee Z')
            ax4.plot(time_axis, net_mag, 'k--', linewidth=2.0, label='Net Magnitude')

            ax4.set_title("Total Net Force on Leg (Knee Frame)", fontsize=16)
            ax4.set_xlabel("Time (s)", fontsize=14)
            ax4.set_ylabel("Force (N)", fontsize=14)
            ax4.grid(True, linestyle='-', alpha=0.3)
            ax4.legend(loc='upper right')

            print(f"Max Net Force (Knee Frame): {np.max(net_mag):.2f} N")
            plt.tight_layout()

        print("Displaying plots...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_data()