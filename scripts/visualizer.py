import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load CSV
# =====================================================

csv_file1 = "data/kp1000_freq360hz.csv"
csv_file2 = "data/kp1000_freq1200hz.csv"  # Ground Truth (highest fidelity)
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Create output directory
os.makedirs("figs", exist_ok=True)

# Extract columns
dt = df1["Time"].iloc[1] - df1["Time"].iloc[0]  # Assuming uniform sampling
step_time = df1.loc[df1["Target Motor Position"] > 0, "Time"].iloc[0] - dt  # Time just before the step occurs
df1 = df1[df1["Time"] >= step_time].copy()
df1["Time"] -= step_time
df1 = df1[df1["Time"] <= 0.1]

t = df1["Time"]

target_pos = df1["Target Motor Position"]
motor_pos = df1["Motor Position"]

motor_vel = df1["Motor Velocity"]

motor_torque = df1["Motor Applied Torque"]
motor_calc_torque = df1["Motor calculated Torque"]

dt_gt = df2["Time"].iloc[1] - df2["Time"].iloc[0]
step_time_gt = df2.loc[df2["Target Motor Position"] > 0, "Time"].iloc[0] - dt_gt
df2 = df2[df2["Time"] >= step_time_gt].copy()
df2["Time"] -= step_time_gt
df2 = df2[df2["Time"] <= 0.1]

t_gt = df2["Time"]
target_pos_gt = df2["Target Motor Position"]
motor_pos_gt = df2["Motor Position"]
motor_vel_gt = df2["Motor Velocity"]
motor_torque_gt = df2["Motor Applied Torque"]
motor_calc_torque_gt = df2["Motor calculated Torque"]

# =====================================================
# Figure 1: Target vs Motor Position
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(t, target_pos, label="Target Position")
plt.plot(t, motor_pos, label="Motor Position")
plt.plot(t_gt, target_pos_gt, "--", label="Ground Truth Target Position")
plt.plot(t_gt, motor_pos_gt, "--", label="Ground Truth Motor Position")
plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.title("Motor Position Tracking")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figs/position_tracking.png", dpi=300)
plt.close()

# =====================================================
# Figure 2: Motor Velocity
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(t, motor_vel, label="Test Motor Velocity")
plt.plot(t_gt, motor_vel_gt, "--", label="Ground Truth Motor Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [rad/s]")
plt.title("Motor Velocity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figs/motor_velocity.png", dpi=300)
plt.close()

# =====================================================
# Figure 3: Torque
# =====================================================

plt.figure(figsize=(10, 5))
plt.plot(t, motor_torque, label="Applied Torque")
# plt.plot(t, motor_calc_torque, label="Calculated Torque")
plt.plot(t_gt, motor_torque_gt, "--", label="Ground Truth Applied Torque")
# plt.plot(t_gt, motor_calc_torque_gt, "--", label="Ground Truth Calculated Torque")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.title("Motor Torque")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figs/motor_torque.png", dpi=300)
plt.close()

print("Figures saved to ./figs/")