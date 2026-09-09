import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "python-stuff/multiple_arrays.npz"
OUTPUT_DIR = "python-stuff/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DT = 1.0 / 90.0
START_TIME = 0.5  # Filter start time in seconds
END_TIME = 3.0  # Filter end time in seconds


def plot_knee_angle_torque():
    try:
        print(f"Loading {FILE_PATH}...")
        loaded_data = np.load(FILE_PATH)

        # ==========================================================
        # Load & Process Data
        # ==========================================================
        # 1. Knee Z Force
        if "skin_forces_knee" in loaded_data:
            knee_forces = loaded_data["skin_forces_knee"]
            net_knee_vector = np.sum(knee_forces, axis=1)
            knee_fz = net_knee_vector[:, 2]
        else:
            raise KeyError("`skin_forces_knee` array not found in dataset.")

        # 2. Ankle Angles
        reference_ankle_angle = loaded_data["ref_angle_data"].flatten()
        ankle_angle_data = loaded_data["ankle_angle_data"].flatten()
        motor_angle_data = loaded_data["motor_angle_data"].flatten()
        actual_ankle_angle = ankle_angle_data + motor_angle_data

        # 3. Motor Torque
        motor_torque_data = loaded_data["motor_torque_data"].flatten()

        # 4. Lower-body Power Consumption}
        if "lower_body_power_data" in loaded_data:
            lower_body_power = loaded_data["lower_body_power_data"].flatten()
        else:
            raise KeyError( "`lower_body_power_data` array not found in dataset." )

        # 4. Time Axis & Time Masking (from START_TIME >= 0.5s)
        num_frames = len(motor_torque_data)
        time_axis = np.linspace(0, num_frames * DT, num_frames)

        mask = (time_axis >= START_TIME) & (time_axis <= END_TIME)
        t_plot = time_axis[mask]
        knee_fz_plot = knee_fz[mask]
        ref_angle_plot = reference_ankle_angle[mask]
        act_angle_plot = actual_ankle_angle[mask]
        torque_plot = motor_torque_data[mask]
        power_plot = lower_body_power[mask]

        # ==========================================================
        # Plot 3-Panel Figure
        # ==========================================================
        fig, axs = plt.subplots(3, 1, figsize=(5*1.1, 4.13333*1.1), sharex=True)

        # Subplot 1: Knee Z Force
        axs[0].plot(
            t_plot,
            knee_fz_plot,
            color="tab:blue",
            linewidth=1.5,
            label="Fuerza sobre Muñón",
        )
        axs[0].set_ylabel("Fuerza (N)")
        axs[0].set_title("Fuerza sobre muñón", fontweight="bold")
        axs[0].grid(True, linestyle="--", alpha=0.6)
        # axs[0].legend(loc="upper right")

        # Subplot 2: Desired Angle vs Actual Angle
        axs[1].plot(
            t_plot,
            ref_angle_plot,
            color="tab:blue",
            linewidth=1.5,
            label="Referencia",
        )
        axs[1].plot(
            t_plot,
            act_angle_plot,
            color="tab:orange",
            linewidth=1.5,
            label="Actual",
        )
        axs[1].set_ylabel("Ángulo (rad)")
        axs[1].set_title("Referencia vs Actual", fontweight="bold")
        axs[1].grid(True, linestyle="--", alpha=0.6)
        axs[1].legend(loc="upper right")

        # Subplot 3: Motor Torque
        axs[2].plot(
            t_plot,
            torque_plot,
            color="tab:green",
            linewidth=1.5,
            label="Motor Torque",
        )
        axs[2].set_xlabel("Tiempo (s)", fontweight="bold")
        axs[2].set_ylabel("Torque (Nm)")
        axs[2].set_title("Torque del Motor", fontweight="bold")
        axs[2].grid(True, linestyle="--", alpha=0.6)
        # axs[2].legend(loc="upper right")

        plt.tight_layout()

        # Save and display
        output_file = os.path.join(
            OUTPUT_DIR, "knee_angle_torque_3panel_from.png"
        )
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

        # ==========================================================
        # Separate Figure: Lower-Body Power Consumption
        # ==========================================================

        power_plot_norm = power_plot
        POWER_EXP_COEFFICIENT = 0.001
        

        fig_power, ax_power = plt.subplots( figsize=(10, 4.5) )
        ax_power.plot( t_plot, power_plot_norm, linewidth=1.5, label="Lower-body power", )
        lower_body_power_exp = np.exp( POWER_EXP_COEFFICIENT * power_plot )
        lower_body_power_exp_plot = lower_body_power_exp*200
        ax_power.plot( t_plot, lower_body_power_exp_plot, linewidth=1.5, label=f"Lower-body exp {POWER_EXP_COEFFICIENT}",)
        ax_power.set_xlabel( "Tiempo (s)", fontweight="bold" )
        ax_power.set_ylabel( "Potencia", fontweight="bold" )
        ax_power.set_title( "Consumo de Potencia — Lower Body", fontweight="bold" )
        ax_power.grid( True, linestyle="--", alpha=0.6 )
        ax_power.legend( loc="upper right" )
        plt.tight_layout()
        power_output_file = os.path.join( OUTPUT_DIR, "lower_body_power.png" )
        fig_power.savefig( power_output_file, dpi=300, bbox_inches="tight" )
        print( f"Power plot saved to: {power_output_file}" )
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_knee_angle_torque()