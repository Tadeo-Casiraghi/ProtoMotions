import mujoco as mj
import numpy as np

# Load the model from an MJCF file (e.g., "my_model.xml")
# Replace "my_model.xml" with your model file path
model = mj.MjModel.from_xml_path("smpl_humanoid9.xml")

# Get total mass by summing all body masses
total_mass = np.sum(model.body_mass)

# Get the gravity vector
gravity_vector = model.opt.gravity

# Calculate the total weight (force vector) W = m * g
# The total weight is a force vector, usually acting vertically
total_weight_vector = total_mass * gravity_vector

# The magnitude of the total weight
total_weight_magnitude = np.linalg.norm(total_weight_vector)

print(f"Total Mass: {total_mass:.3f} kg")
print(f"Gravity Vector: {gravity_vector}")
print(f"Total Weight Vector: {total_weight_vector} N")
print(f"Total Weight Magnitude: {total_weight_magnitude:.3f} N")
