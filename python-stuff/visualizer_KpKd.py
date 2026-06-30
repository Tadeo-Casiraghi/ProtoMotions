import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "python-stuff/multiple_arrays.npz"   # Ensure path is correct
FILE_PATH2 = "python-stuff/sim_torque.txt"       # CSV file from simulator logger
DT = 1.0 / 30.0
# ---------------------


def action_to_impedance_targets(raw_theta, raw_kp, raw_kd):
    # Map [-1, 1] -> [-pi, pi]
    desired_angle = raw_theta * 3.14

    # Map [-1, 1] -> [0, 2000]
    kp_phys = raw_kp * 1000.0 + 1000.0

    # Map [-1, 1] -> [0, 10]
    kd_phys = raw_kd * 5.0 + 5.0

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

        motor_angle_data = loaded_data["motor_angle_data"].flatten()
        motor_velocity_data = loaded_data["motor_velocity_data"].flatten()
        motor_torque_data = loaded_data["motor_torque_data"].flatten()

        # ==========================================================
        # Reconstruct torque from NPZ
        # ==========================================================
        q_prev_est = motor_angle_data - motor_velocity_data / 120.0

        angle_error = desired_angle_data - q_prev_est

        torque_from_kp = kp_data * angle_error
        torque_from_kd = kd_data * (-motor_velocity_data)

        total_torque = torque_from_kp + torque_from_kd

        # ==========================================================
        # Load simulator CSV
        # Expected header:
        # kp,kd,theta_des,theta_cur,vel_cur,torque_calc,torque_applied
        # ==========================================================
        print(f"Loading {FILE_PATH2}...")

        sim_data = np.genfromtxt(
            FILE_PATH2,
            delimiter=",",
            names=True,
            dtype=float,
        )

        sim_kp = sim_data["kp"]
        sim_kd = sim_data["kd"]

        sim_theta_des = sim_data["theta_des"]
        sim_theta_cur = sim_data["theta_cur"]

        sim_vel_cur = sim_data["vel_cur"]

        sim_torque_calc = sim_data["torque_calc"]
        sim_torque_applied = sim_data["torque_applied"]

        # ==========================================================
        # FIGURE 1: KP
        # ==========================================================
        plt.figure(figsize=(12, 6))
        plt.plot(sim_kp, label="Kp", color="blue")
        # plt.plot(np.diff(sim_kp)/(1/360), label="Kp derivative", color="red")
        plt.title("Kp")
        plt.xlabel("Sample")
        plt.ylabel("Kp")
        plt.grid(True)
        plt.legend()

        # ==========================================================
        # FIGURE 2: KD
        # ==========================================================
        plt.figure(figsize=(12, 6))
        plt.plot(sim_kd, label="Kd", color="orange")
        # plt.plot(np.diff(sim_kd)/(1/360), label="Kd derivative", color="red")
        plt.title("Kd")
        plt.xlabel("Sample")
        plt.ylabel("Kd")
        plt.grid(True)
        plt.legend()

        # ==========================================================
        # FIGURE 3: Calculated Torque vs Applied Torque
        # ==========================================================
        plt.figure(figsize=(12, 6))
        plt.plot(sim_torque_calc, label="Calculated Torque")
        plt.plot(sim_torque_applied, label="Applied Torque")
        plt.title("Calculated Torque vs Applied Torque")
        plt.xlabel("Sample")
        plt.ylabel("Torque")
        plt.grid(True)
        plt.legend()

        # ==========================================================
        # FIGURE 4: Desired Theta vs Current Theta
        # ==========================================================
        plt.figure(figsize=(12, 6))
        plt.plot(sim_theta_des, label="Theta Desired")
        # plt.plot(np.diff(sim_theta_des)/(1/360), label="Theta Desired Derivative", color="red")
        plt.plot(sim_theta_cur, label="Theta Current")
        plt.title("Theta Desired vs Theta Current")
        plt.xlabel("Sample")
        plt.ylabel("Angle (rad)")
        plt.grid(True)
        plt.legend()

        # ==========================================================
        # Optional: Original NPZ plots for comparison
        # ==========================================================

        # Kp and Kd from policy
        plt.figure(figsize=(12, 6))
        plt.plot(kp_data, label="Policy Kp")
        plt.plot(kd_data, label="Policy Kd")
        plt.title("Policy Kp and Kd")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        # Desired vs actual angle from NPZ
        plt.figure(figsize=(12, 6))
        plt.plot(desired_angle_data, label="Desired Angle")
        plt.plot(motor_angle_data, label="Motor Angle")
        plt.title("Policy Desired Angle vs Motor Angle")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.grid(True)
        plt.legend()

        # Torque comparison from NPZ
        plt.figure(figsize=(12, 6))
        plt.plot(motor_torque_data, label="Motor Torque")
        plt.plot(
            total_torque,
            label="Reconstructed Torque (Kp + Kd)",
            linestyle="--",
        )
        plt.title("Motor Torque vs Reconstructed Torque")
        plt.xlabel("Time Step")
        plt.ylabel("Torque")
        plt.grid(True)
        plt.legend()

        print("Displaying plots...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_data()