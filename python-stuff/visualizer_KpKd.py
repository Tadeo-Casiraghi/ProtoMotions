import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "python-stuff/multiple_arrays.npz"   # Ensure path is correct
FILE_PATH2 = "python-stuff/sim_torque.txt"       # CSV file from simulator logger
OUTPUT_DIR = "python-stuff/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DT = 1.0 / 90.0

upto = 600
# ---------------------


def action_to_impedance_targets(raw_theta, raw_kp, raw_kd):
    # Map [-1, 1] -> [-pi, pi]
    desired_angle = raw_theta * 3.14

    # Map [-1, 1] -> [0, 2000]
    kp_phys = raw_kp * 250.0 + 250.0

    # Map [-1, 1] -> [0, 10]
    kd_phys = raw_kd * 2.5 + 2.5

    return kp_phys, kd_phys, desired_angle


def plot_data():
    try:
        # ==========================================================
        # Load NPZ file
        # ==========================================================
        print(f"Loading {FILE_PATH}...")
        loaded_data = np.load(FILE_PATH)

        kp_data = loaded_data["kp_data"].flatten()
        kd_data = loaded_data["kd_data"].flatten()
        desired_angle_data = loaded_data["desired_angle_data"].flatten()

        kp_data, kd_data, desired_angle_data = action_to_impedance_targets(
            desired_angle_data,
            kp_data,
            kd_data,
        )

        reference_ankle_angle = loaded_data["ref_angle_data"].flatten()
        ankle_angle_data = loaded_data["ankle_angle_data"].flatten()

        motor_angle_data = loaded_data["motor_angle_data"].flatten()
        motor_velocity_data = loaded_data["motor_velocity_data"].flatten()
        motor_torque_data = loaded_data["motor_torque_data"].flatten()

        time_line = np.arange(0,20,DT)

        # ==========================================================
        # Reconstruct torque from NPZ
        # ==========================================================
        q_prev_est = motor_angle_data - motor_velocity_data / 360.0

        angle_error = desired_angle_data - q_prev_est

        torque_from_kp = kp_data * angle_error
        torque_from_kd = kd_data * motor_velocity_data

        total_torque = torque_from_kp - torque_from_kd

        # ==========================================================
        # Optional: Original NPZ plots for comparison
        # ==========================================================

        # Kp and Kd from policy
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Left y-axis for Kp
        color1 = "tab:blue"
        ax1.plot(time_line[:upto], kp_data[:upto], color=color1, label="Policy Kp")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Kp", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True)

        # Right y-axis for Kd
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.plot(time_line[:upto], kd_data[:upto], color=color2, label="Policy Kd")
        ax2.set_ylabel("Kd", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 6)  # Set limits for Kd

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title("Policy Kp and Kd")

        fig.savefig(
            os.path.join(OUTPUT_DIR, "policy_kp_kd.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # Desired vs actual angle from NPZ
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(time_line[:upto], desired_angle_data[:upto], label="Desired Angle")
        plt.plot(time_line[:upto], motor_angle_data[:upto], label="Motor Angle")
        plt.title("Policy Desired Angle vs Motor Angle")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.grid(True)
        plt.legend()

        fig2.savefig(
            os.path.join(OUTPUT_DIR, "desired_vs_motor_angle.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # Torque comparison from NPZ
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(time_line[:upto], motor_torque_data[:upto], label="Motor Torque")
        plt.plot(time_line[:upto], 
            total_torque[:upto],
            label="Reconstructed Torque (Kp + Kd)",
            linestyle="--",
        )
        plt.title("Motor Torque vs Reconstructed Torque")
        plt.xlabel("Time Step")
        plt.ylabel("Torque")
        plt.grid(True)
        plt.legend()

        fig3.savefig(
            os.path.join(OUTPUT_DIR, "motor_vs_reconstructed_torque.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # ==========================================================
        # FIGURE: Reference Ankle angle vs Actual Ankle angle
        # ==========================================================

        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(time_line[:upto], reference_ankle_angle[:upto], label="Desired Angle")
        plt.plot(time_line[:upto], ankle_angle_data[:upto], label="Ankle Angle")
        plt.plot(time_line[:upto], motor_angle_data[:upto], label="Motor Angle")
        plt.plot(time_line[:upto], motor_angle_data[:upto] + ankle_angle_data[:upto], label="Ankle + Motor Angle")
        plt.title("Reference Ankle Angle vs Actual Ankle Angle")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.grid(True)
        plt.legend()

        fig4.savefig(
            os.path.join(OUTPUT_DIR, "reference_vs_ankle_angle.png"),
            dpi=300,
            bbox_inches="tight",
        )


        # ==========================================================
        # FIGURE: 3 in 1
        # ========================================================== 

        fig5, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        
        # torque

        axs[0].plot(time_line[:upto], motor_torque_data[:upto], label="Motor Torque")
        axs[0].plot(time_line[:upto], total_torque[:upto], label="Reconstructed Torque (Kp + Kd)", linestyle="--")
        axs[0].set_ylabel("Torque")
        axs[0].set_title("Motor Torque vs Reconstructed Torque")
        axs[0].grid(True)
        axs[0].legend()

        # angles
        axs[1].plot(time_line[:upto], reference_ankle_angle[:upto], label="Desired Angle")
        axs[1].plot(time_line[:upto], ankle_angle_data[:upto] + motor_angle_data[:upto], label="Combined Angle")
        axs[1].set_ylabel("Angle (rad)")
        axs[1].set_title("Reference Ankle Angle vs Actual Ankle Angle")
        axs[1].grid(True)
        axs[1].legend()

        # Kp and Kd
        axs[2].plot(time_line[:upto], kp_data[:upto], label="Policy Kp", color="tab:blue")

        ax_temp = axs[2].twinx()
        ax_temp.plot(time_line[:upto], kd_data[:upto], label="Policy Kd", color="tab:orange")
        ax_temp.set_ylabel("Kd", color="tab:orange")
        axs[2].set_ylabel("Kp / Kd")
        axs[2].set_title("Policy Kp and Kd")
        axs[2].grid(True)
        axs[2].legend()

        plt.xlabel("Time Step")

        plt.tight_layout()
        fig5.savefig(
            os.path.join(OUTPUT_DIR, "combined_plot.png"),
            dpi=300,
            bbox_inches="tight",
        )


        # ==========================================================
        # FIGURE: velocity
        # ========================================================== 

        # Desired vs actual angle from NPZ
        fig6 = plt.figure(figsize=(12, 6))
        plt.plot(time_line[:upto], motor_velocity_data[:upto], label="Motor Velocity")
        plt.title("Motor Vel")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity (rad/s)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        fig6.savefig(
            os.path.join(OUTPUT_DIR, "velocity.png"),
            dpi=300,
            bbox_inches="tight",
        )

        print("Displaying plots...")

        # plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_data()