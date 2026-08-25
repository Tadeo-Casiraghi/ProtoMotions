import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILE_PATH = "python-stuff/multiple_arrays.npz"   # Ensure path is correct
FILE_PATH2 = "python-stuff/sim_torque.txt"       # CSV file from simulator logger
OUTPUT_DIR = "python-stuff/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        # print(f"Loading {FILE_PATH2}...")

        # sim_data = np.genfromtxt(
        #     FILE_PATH2,
        #     delimiter=",",
        #     names=True,
        #     dtype=float,
        # )

        # sim_kp = sim_data["kp"]
        # sim_kd = sim_data["kd"]

        # sim_theta_des = sim_data["theta_des"]
        # sim_theta_cur = sim_data["theta_cur"]

        # sim_vel_cur = sim_data["vel_cur"]

        # sim_torque_calc = sim_data["torque_calc"]
        # sim_torque_applied = sim_data["torque_applied"]

        # # ==========================================================
        # # FIGURE 1: KP
        # # ==========================================================
        # plt.figure(figsize=(12, 6))
        # plt.plot(sim_kp, label="Kp", color="blue")
        # # plt.plot(np.diff(sim_kp)/(1/360), label="Kp derivative", color="red")
        # plt.title("Kp")
        # plt.xlabel("Sample")
        # plt.ylabel("Kp")
        # plt.grid(True)
        # plt.legend()

        # # ==========================================================
        # # FIGURE 2: KD
        # # ==========================================================
        # plt.figure(figsize=(12, 6))
        # plt.plot(sim_kd, label="Kd", color="orange")
        # # plt.plot(np.diff(sim_kd)/(1/360), label="Kd derivative", color="red")
        # plt.title("Kd")
        # plt.xlabel("Sample")
        # plt.ylabel("Kd")
        # plt.grid(True)
        # plt.legend()

        # # ==========================================================
        # # FIGURE 3: Calculated Torque vs Applied Torque
        # # ==========================================================
        # plt.figure(figsize=(12, 6))
        # plt.plot(sim_torque_calc, label="Calculated Torque")
        # plt.plot(sim_torque_applied, label="Applied Torque")
        # plt.title("Calculated Torque vs Applied Torque")
        # plt.xlabel("Sample")
        # plt.ylabel("Torque")
        # plt.grid(True)
        # plt.legend()

        # # ==========================================================
        # # FIGURE 4: Desired Theta vs Current Theta
        # # ==========================================================
        # plt.figure(figsize=(12, 6))
        # plt.plot(sim_theta_des, label="Theta Desired")
        # # plt.plot(np.diff(sim_theta_des)/(1/360), label="Theta Desired Derivative", color="red")
        # plt.plot(sim_theta_cur, label="Theta Current")
        # plt.title("Theta Desired vs Theta Current")
        # plt.xlabel("Sample")
        # plt.ylabel("Angle (rad)")
        # plt.grid(True)
        # plt.legend()

        # ==========================================================
        # Optional: Original NPZ plots for comparison
        # ==========================================================

        # Kp and Kd from policy
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Left y-axis for Kp
        color1 = "tab:blue"
        ax1.plot(kp_data[:100], color=color1, label="Policy Kp")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Kp", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True)

        # Right y-axis for Kd
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.plot(kd_data[:100], color=color2, label="Policy Kd")
        ax2.set_ylabel("Kd", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

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
        plt.plot(desired_angle_data[:100], label="Desired Angle")
        plt.plot(motor_angle_data[:100], label="Motor Angle")
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
        plt.plot(motor_torque_data[:100], label="Motor Torque")
        plt.plot(
            total_torque[:100],
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

        # # ==========================================================
        # # Time axis and plotting mask
        # # ==========================================================
        # DT = 1/360
        # MAX_PLOT_TIME = 10
        # time = np.arange(len(sim_kp)) * DT

        # if MAX_PLOT_TIME is None:
        #     mask = np.ones_like(time, dtype=bool)
        # else:
        #     mask = time <= MAX_PLOT_TIME

        # # ==========================================================
        # # FIGURE: Policy outputs over time
        # # ==========================================================
        # fig, axs = plt.subplots(
        #     3, 1,
        #     figsize=(12, 8),
        #     sharex=True
        # )

        # times = []
        # kps = []
        # kds = []
        # thetas = []
        # current = 0
        # time0 = 0

        # for t, kp, kd, th in zip(time, sim_kp, sim_kd, sim_theta_des):
        #     if times == []:
        #         times.append([])
        #         kps.append([])
        #         kds.append([])
        #         thetas.append([])
            
        #     elif abs(kds[current][-1] - kd) > 2:
        #         current += 1
        #         time0 = t
        #         times.append([])
        #         kps.append([])
        #         kds.append([])
        #         thetas.append([])

        #     times[current].append(t-time0)
        #     kps[current].append(kp)
        #     kds[current].append(kd)
        #     thetas[current].append(th)
            
        # for i in range(len(times)):
        #     time = np.array(times[i])
        #     kp = np.array(kps[i])
        #     kd = np.array(kds[i])
        #     theta = np.array(thetas[i])

        #     tim1 = 1.079
        #     mask1 = (time > 1.079) & (time < 2.549)
        #     tim2 = 2.549
        #     mask2 = (time > tim2)
        #     t1 = time[mask1]-tim1
        #     mask3 = t1 < 1.1
        #     t1 = t1[mask3]
        #     t2 = time[mask2]-tim2
        #     mask4 = t2 < 1.1
        #     t2 = t2[mask4]
        #     kp1 = kp[mask1]
        #     kp1 = kp1[mask3]
        #     kp2 = kp[mask2]
        #     kp2 = kp2[mask4]

        #     kd1 = kd[mask1]
        #     kd1 = kd1[mask3]
        #     kd2 = kd[mask2]
        #     kd2 = kd2[mask4]

        #     th1 = theta[mask1]
        #     th1 = th1[mask3]
        #     th2 = theta[mask2]
        #     th2 = th2[mask4]

            

        #     axs[0].plot(t1, kp1, color="tab:blue", alpha=0.5)
        #     axs[0].plot(t2, kp2, color="tab:blue", alpha=0.5)

        #     axs[1].plot(t1, kd1, color="tab:orange", alpha=0.5)
        #     axs[1].plot(t2, kd2, color="tab:orange", alpha=0.5)

        #     axs[2].plot(t1, th1, color="tab:green", alpha=0.5)
        #     axs[2].plot(t2, th2, color="tab:green", alpha=0.5)
            
        
        # axs[0].set_ylabel("Kp")
        # axs[0].set_title("Policy Kp")
        # axs[0].grid(True)

        # axs[1].set_ylabel("Kd")
        # axs[1].set_title("Policy Kd")
        # # axs[1].axvline(1344)
        # axs[1].grid(True)

        # axs[2].set_ylabel("Theta (rad)")
        # axs[2].set_xlabel("Time (s)")
        # axs[2].set_title("Policy Theta")
        # axs[2].grid(True)

        # plt.tight_layout()

        print("Displaying plots...")

        # plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_data()