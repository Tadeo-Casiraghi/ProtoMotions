import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control as ctrl
import re

def extract_actuators(lines):
    actuators = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Detect actuator start
        if line.startswith("Actuator:"):
            name = line.split(":")[1].strip()
            data = {}

            i += 1
            # Read actuator properties
            while i < len(lines) and lines[i].strip() != "":
                prop_line = lines[i].strip()

                if ":" in prop_line:
                    key, value = prop_line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Handle None values
                    if value == "None":
                        data[key] = None
                    else:
                        data[key] = float(value)

                i += 1

            actuators[name] = data

        i += 1

    return actuators


def extract_rigid_bodies(lines):
    bodies = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Body"):
            # Example: Body 2: R_Ankle
            name = line.split(":")[1].strip()
            data = {}

            i += 1
            while i < len(lines) and lines[i].strip() != "":
                l = lines[i].strip()

                if l.startswith("Mass:"):
                    mass = float(l.split(":")[1].replace("kg", "").strip())
                    data["mass"] = mass

                elif "tensor([" in l:
                    # Extract tensor values
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                    tensor = np.array(list(map(float, numbers)))
                    data["inertia_tensor"] = tensor.reshape(3, 3)

                i += 1

            bodies[name] = data

        i += 1

    return bodies


def parallel_axis(I_com, m, r):
    r = np.array(r)
    return I_com + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))




# Read file and extract data
test_number = "Server"
test_number = 9

with open(f"test{test_number}/data.txt", "r") as f:
    lines = f.readlines()

actuators = extract_actuators(lines)

motor = actuators["Motor"]
spring = actuators["R_Ankle_y"]

print("Motor:", motor)
print("Spring:", spring)

bodies = extract_rigid_bodies(lines)

ankle = bodies["R_Ankle"]
toe = bodies["R_Toe"]
motor_body = bodies["R_Ankle_Motor"]

print(ankle)
print(toe)

# Positions from USD
r_ankle = np.array([0.0, 0.0, 0.0])
r_toe = np.array([0.0660, -0.3389, -0.8984])

I_ankle = ankle["inertia_tensor"]
m_ankle = ankle["mass"]

I_toe = toe["inertia_tensor"]
m_toe = toe["mass"]

I_ankle_joint = parallel_axis(I_ankle, m_ankle, r_ankle)
I_toe_joint = parallel_axis(I_toe, m_toe, r_toe)

I_total = I_ankle_joint + I_toe_joint

Jl = I_total[2,2]*0.36
print(I_total)



# Read step data
data = pd.read_csv(f"test{test_number}/step_response_data.csv")

time_data = data["Time"].values
target_data = data["Target Motor Position"].values
actual_data = data["Ankle Position"].values + data["Motor Position"].values

step_time = time_data[np.where(target_data > 0)[0][0]]

# Transfer Function
# if Kp in data as a column, use it; otherwise, use the value from the motor data
if "Kp" in data.columns:
    Kp = data["Kp"].values[0]
else:
    Kp = motor["stiffness"]

if "Kd" in data.columns:
    Kd = data["Kd"].values[0]
else:
    Kd = motor["damping"]
Jm = motor["armature"] + 1e-4
k = spring["stiffness"]
b = spring["damping"]

# Tf = ctrl.TransferFunction([Kp*b, Kp*k], [Jm*Jl, (Jm*b + Kd*Jl + Jl*b), (Jm*k + Kd*b + Kp*Jl+k*Jl), (Kd*k+Kp*b), Kp*k])


# t_out, y_out = ctrl.forced_response(Tf, T=time_data, U=target_data)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(time_data, target_data, label="Target Position (rad)", linestyle="--")
# plt.plot(time_data, actual_data, label="Actual Position (rad)")
# plt.plot(t_out+1/240, y_out, label="Step Response")
# plt.title("Step Response of Joint 1")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (rad)")
# plt.xlim(2, 3.5)
# plt.legend()
# plt.grid()
# plt.show()

# =====================================================================
# INSERT HZ VALUE HERE
# Try changing this to 120, 60, or 30 to see the tracking degrade
# =====================================================================
Hz = 120 
T = 1.0 / Hz 

# Define continuous laplace variable 's'
s = ctrl.tf('s')

# Open Loop Plant Subsystems (Derived from coupled physical equations)
# Common Denominator: Jm*Jl*s^4 + (Jm+Jl)*b*s^3 + (Jm+Jl)*k*s^2
den_plant = Jm*Jl*s**4 + (Jm + Jl)*b*s**3 + (Jm + Jl)*k*s**2
P_motor = (Jl*s**2 + b*s + k) / den_plant
P_load = (b*s + k) / den_plant

# Discrete Feedback Controller mapped to Continuous domain via Bilinear Transform
# The backward difference derivative becomes an ideal derivative with a low-pass filter
C_fb = Kp + Kd * (s / (1 + (T / 2) * s))

# Complete Closed-Loop Transfer Functions
Tf_motor = (Kp * P_motor) / (1 + C_fb * P_motor)
Tf_load = (Kp * P_load) / (1 + C_fb * P_motor)
# =====================================================================

# Simulate Responses
t_out_m, y_out_m = ctrl.forced_response(Tf_motor, T=time_data, U=target_data)
t_out_l, y_out_l = ctrl.forced_response(Tf_load, T=time_data, U=target_data)

# --- Plotting Subplots ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. Motor Tracking Plot
ax1.plot(time_data, target_data, label="Target Motor Pos (rad)", linestyle="--", color="black")
ax1.plot(time_data, actual_data, label="Actual Motor Pos (Data)", color="blue", alpha=0.7)
ax1.plot(t_out_m + 1/240, y_out_m, label=f"Simulated Motor Pos ({Hz}Hz)", linestyle="-.", color="cyan")
ax1.set_title("Motor Tracking Performance")
ax1.set_ylabel("Position (rad)")
ax1.legend()
ax1.grid(True)

# 2. Load (Ankle Joint) Plot
ax2.plot(time_data, actual_data, label="Actual Ankle Pos (Data)", color="green", alpha=0.7)
ax2.plot(t_out_l + 1/240, y_out_l, label=f"Simulated Ankle Pos ({Hz}Hz)", linestyle="-.", color="lime")
ax2.set_title("Load (Ankle Joint) Dynamic Response")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (rad)")
ax2.set_xlim(2, 3.5)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()