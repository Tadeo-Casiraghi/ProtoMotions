import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "python-stuff/multiple_arrays.npz" # Ensure path is correct
DT = 1.0 / 30.0
# ---------------------

def action_to_impedance_targets(
        raw_theta,
        raw_kp, 
        raw_kd):
        # Map [-1, 1] -> [-pi, pi]
        desired_angle = raw_theta * 3.14 + 0.0
        
        # Map [-1, 1] -> [0, 1000]
        kp_phys       = raw_kp * 1000.0 + 1000.0
        
        # Map [-1, 1] -> [0, 10]
        kd_phys       = raw_kd * 5.0 + 5.0
        
        return kp_phys, kd_phys, desired_angle

def plot_data():
    try:
        print(f"Loading {FILE_PATH}...")
        loaded_data = np.load(FILE_PATH)
        
        # --- Load Kinematics ---
        kp_data = loaded_data['kp_data'].flatten()
        kd_data = loaded_data['kd_data'].flatten()
        desired_angle_data = loaded_data['desired_angle_data'].flatten()

        kp_data, kd_data, desired_angle_data = action_to_impedance_targets(
            desired_angle_data,
            kp_data,
            kd_data
        )

        motor_angle_data = loaded_data['motor_angle_data'].flatten()
        motor_velocity_data = loaded_data['motor_velocity_data'].flatten()
        motor_torque_data = loaded_data['motor_torque_data'].flatten()


        # --- Plot Kp and Kd ---
        plt.figure(figsize=(12, 10))
        plt.plot(kp_data, label='Kp', color='blue')
        plt.plot(kd_data, label='Kd', color='orange')
        plt.title('Kp and Kd over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        # --- Plot Desired Angle and Motor Angle ---
        plt.figure(figsize=(12, 10))
        plt.plot(desired_angle_data, label='Desired Angle', color='green')
        plt.plot(motor_angle_data, label='Motor Angle', color='red')
        plt.title('Desired Angle vs Motor Angle')
        plt.xlabel('Time Steps')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid()

        # --- Plot Motor Torque ---
        q_prev_est = motor_angle_data - motor_velocity_data * 1/120
        # calculate torque from Kp, Kd, and angle error
        angle_error = desired_angle_data - q_prev_est
        torque_from_kp = kp_data * angle_error
        torque_from_kd = kd_data * (-motor_velocity_data)
        total_torque = torque_from_kp + torque_from_kd


        plt.figure(figsize=(12, 10))
        plt.plot(motor_torque_data, label='Motor Torque', color='purple')
        plt.plot(total_torque, label='Calculated Torque (Kp + Kd)', color='cyan', linestyle='--')
        plt.title('Motor Torque over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid()

        print("Displaying plots...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_data()