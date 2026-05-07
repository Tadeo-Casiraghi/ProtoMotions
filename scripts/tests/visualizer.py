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
test_number = 5

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
Kp = motor["stiffness"]
Kd = motor["damping"]
Jm = motor["armature"] + 1e-4
k = spring["stiffness"]
b = spring["damping"]

Tf = ctrl.TransferFunction([Kp*b, Kp*k], [Jm*Jl, (Jm*b + Kd*Jl + Jl*b), (Jm*k + Kd*b + Kp*Jl+k*Jl), (Kd*k+Kp*b), Kp*k])


t_out, y_out = ctrl.forced_response(Tf, T=time_data, U=target_data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_data, target_data, label="Target Position (rad)", linestyle="--")
plt.plot(time_data, actual_data, label="Actual Position (rad)")
plt.plot(t_out+1/240, y_out, label="Step Response")
plt.title("Step Response of Joint 1")
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.xlim(2, 3)
plt.legend()
plt.grid()
plt.show()