import argparse
import csv
import os
from isaaclab.app import AppLauncher
import sys

# ---------------------------------------------------------------------------
# Parse sweep parameters BEFORE launching the app.
#
# --kp / --kd       gains for the manual PD torque law applied to "Motor"
# --freq            physics simulation frequency in Hz  (physics_dt = 1/freq)
# --control-hz      how often the control law recomputes torque. Decimation is
#                   derived from this so the *control update rate* stays fixed
#                   across the dt sweep -- otherwise changing physics_dt would
#                   also silently change how often the controller runs, and
#                   you'd be testing two things at once instead of isolating
#                   physics fidelity.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--kp", type=float, default=2000.0)
parser.add_argument("--kd", type=float, default=5.0)
parser.add_argument("--freq", type=float, default=120.0, help="Physics sim frequency in Hz")
parser.add_argument("--control-hz", type=float, default=60.0, help="Control loop update rate in Hz")
cli_args = parser.parse_args()

Kp = cli_args.kp
Kd = cli_args.kd
physics_dt = 1.0 / cli_args.freq
# How many physics steps occur per control update, derived so control period
# (decimation * physics_dt) stays ~constant across the sweep.
decimation = max(1, round((1.0 / cli_args.control_hz) / physics_dt))

# 1. Start the Isaac Sim application
launcher = AppLauncher(headless=True)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

def save_robot_configuration(robot: Articulation, filename="data.txt"):
    with open(filename, "w") as f:

        f.write("========== ROBOT CONFIGURATION ==========\n\n")

        # -------------------------------------------------
        # JOINT INFORMATION
        # -------------------------------------------------
        f.write("JOINTS\n")
        f.write("----------------------------------------\n")

        for joint_id, joint_name in enumerate(robot.joint_names):

            f.write(f"Joint {joint_id}: {joint_name}\n")

            # Joint limits
            try:
                lower = robot.data.soft_joint_pos_limits[0, joint_id, 0].item()
                upper = robot.data.soft_joint_pos_limits[0, joint_id, 1].item()

                f.write(f"  Position Limits: [{lower:.4f}, {upper:.4f}]\n")
            except:
                pass

            # Velocity limits
            try:
                vel_limit = robot.data.joint_vel_limits[0, joint_id].item()
                f.write(f"  Velocity Limit: {vel_limit:.4f}\n")
            except:
                pass

            # Effort limits
            try:
                effort_limit = robot.data.joint_effort_limits[0, joint_id].item()
                f.write(f"  Effort Limit: {effort_limit:.4f}\n")
            except:
                pass

            f.write("\n")

        # -------------------------------------------------
        # ACTUATOR INFORMATION
        # -------------------------------------------------
        f.write("\nACTUATORS\n")
        f.write("----------------------------------------\n")

        for actuator_name, actuator in robot.actuators.items():

            f.write(f"Actuator: {actuator_name}\n")

            cfg = actuator.cfg

            attrs = [
                "stiffness",
                "damping",
                "armature",
                "friction",
                "effort_limit",
                "velocity_limit",
            ]

            for attr in attrs:
                if hasattr(cfg, attr):
                    f.write(f"  {attr}: {getattr(cfg, attr)}\n")

            f.write("\n")

        # -------------------------------------------------
        # BODY INFORMATION
        # -------------------------------------------------
        f.write("\nRIGID BODIES\n")
        f.write("----------------------------------------\n")

        for body_id, body_name in enumerate(robot.body_names):

            f.write(f"Body {body_id}: {body_name}\n")

            # Mass
            try:
                mass = robot.root_physx_view.get_masses()[0, body_id]
                f.write(f"  Mass: {mass:.6f} kg\n")
            except:
                pass

            # Inertia tensor
            try:
                inertia = robot.root_physx_view.get_inertias()[0, body_id]

                f.write("  Inertia Tensor:\n")
                f.write(f"    {inertia}\n")

            except:
                pass

            f.write("\n")

    print(f"Saved robot configuration to {filename}")


Theta = 0.0

@configclass
class ProstheticTestSceneCfg(InteractiveSceneCfg):

    prosthetic_leg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ProstheticLeg",
        spawn=sim_utils.UsdFileCfg(
            usd_path="prosthetic_ankle_test.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True, 
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "R_Ankle_y": ImplicitActuatorCfg(
                joint_names_expr=["R_Ankle_y.*"],
                stiffness=500.0,
                damping=50.0,
                effort_limit=500.0,
                velocity_limit=100.0,
                armature= 0.02,
                friction=0.0
            ),
            "Motor": ImplicitActuatorCfg(
                joint_names_expr=["Motor.*"],
                stiffness=0.0,         # Adjust to tune response
                damping=0.0,            # Adjust to tune response
                armature=0.001,              # Adjust to tune response
                effort_limit=2000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=50.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
        }
    )

def main():
    os.makedirs("data", exist_ok=True)

    scene_cfg = ProstheticTestSceneCfg(num_envs=1, env_spacing=2.0)

    sim_cfg = sim_utils.SimulationCfg(
        dt=physics_dt,
        gravity=(0.0, 0.0, 0.0),
    )

    # Create PhysX config
    sim_cfg.physx = sim_utils.PhysxCfg()

    # Now set fields AFTER creation
    sim_cfg.physx.max_depenetration_velocity = 1.0
    sim_cfg.physx.enable_external_forces_every_iteration=True

    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot: Articulation = scene["prosthetic_leg"]

    save_robot_configuration(robot, filename=os.path.join("data", "robot_configuration.txt"))

    # Identify joint to test
    test_joint_name = "Motor"
    motor_idx = robot.find_joints(test_joint_name)[0]

    joint_idx = robot.find_joints("R_Ankle_y")[0]

    # Data collection lists
    time_history = []
    target_history = []
    actual_pos_history = []

    motor_vel_history = []
    motor_torque_history = []
    motor_desired_torque_history = []

    ankle_pos_history = []
    ankle_vel_history = []
    ankle_torque_history = []
    ankle_desired_torque_history = []

    Kp_history = []
    Kd_history = []

    current_time = 0.0

    pi = 3.14159265359
    torque_old = 0.0
    theta_old = 0.0
    # Filtro para el Target (fc = 20) usando mapeo exacto
    fc_target = 20
    alpha_params = 1.0 - np.exp(-2.0 * pi * fc_target * physics_dt)

    # Filtro para el Torque (fc = 100) usando mapeo exacto
    fc_torque = 100
    alpha = 1.0 - np.exp(-2.0 * pi * fc_torque * physics_dt)

    # Step Disturbance Parameters
    initial_target = theta_old
    step_target = 0.6      # New target position (radians)
    disturbance_applied = False
    time_since_disturbance = 0.0
    post_disturbance_duration = 5.0 # Record for 5 seconds AFTER the step is applied

    velocity_threshold = 0.1 # Threshold to consider the leg "stable"

    print(f"Kp={Kp}  Kd={Kd}  physics_dt={physics_dt:.6f}s ({cli_args.freq:.0f} Hz)  decimation={decimation}")
    print("Initializing Simulation... Waiting for leg to stabilize.")
    target_positions = torch.zeros((1, robot.num_joints), device=robot.device)

    step_count = 0
    while launcher.app.is_running():
        current_vel = robot.data.joint_vel[0, motor_idx].item()
        current_pos = robot.data.joint_pos[0, motor_idx].item()
        current_cmp_torque = robot.data.computed_torque[0, motor_idx].item()
        current_torque = robot.data.applied_torque[0, motor_idx].item()

        current_vel_ankle = robot.data.joint_vel[0, joint_idx].item()
        current_pos_ankle = robot.data.joint_pos[0, joint_idx].item()
        current_cmp_torque_ankle = robot.data.computed_torque[0, joint_idx].item()
        current_torque_ankle = robot.data.applied_torque[0, joint_idx].item()

        if step_count % decimation == 0:
            # 1. Aplicar el escalón en un tiempo fijo y determinista (ej: 2.5 segundos)
            if not disturbance_applied:
                if current_time >= 3.0: # <--- CAMBIO AQUÍ: Tiempo fijo, sin mirar velocidad
                    print(f"Leg stabilized at t={current_time:.4f}s. Applying Step Disturbance!")
                    disturbance_applied = True
                    current_target = step_target
                else:
                    current_target = initial_target
            else:
                current_target = step_target
                time_since_disturbance += sim_cfg.dt * decimation

        ct = theta_old + alpha_params * (current_target - theta_old)
        theta_old = ct
        torque_desired = Kp * (ct - current_pos) - Kd * current_vel
        torque = torque_old + alpha * (torque_desired - torque_old)
        torque_old = torque


        target_positions[:, motor_idx] = 0.0 # Keep the motor target at zero to let the disturbance cause the movement, or set to current_target for a more traditional PD control approach
        robot.set_joint_position_target(target_positions)
        torque_targets = torch.zeros_like(target_positions)
        torque_targets[:, motor_idx] = torque
        robot.set_joint_effort_target(torque_targets)

        scene.write_data_to_sim()

        # 2. Step physics
        sim.step()
        scene.update(sim.get_physics_dt())

        # 3. Record Data (only start recording right before the disturbance for clean data, or record all)
        time_history.append(current_time)
        target_history.append(ct)
        actual_pos_history.append(current_pos)
        motor_vel_history.append(current_vel)
        motor_torque_history.append(current_torque)
        motor_desired_torque_history.append(current_cmp_torque)

        ankle_pos_history.append(current_pos_ankle)
        ankle_vel_history.append(current_vel_ankle)
        ankle_torque_history.append(current_torque_ankle)
        ankle_desired_torque_history.append(current_cmp_torque_ankle)

        Kp_history.append(Kp)
        Kd_history.append(Kd)

        current_time += sim_cfg.dt
        step_count += 1

        # End simulation after recording the post-disturbance response
        if disturbance_applied and time_since_disturbance >= post_disturbance_duration:
            print("Finished recording step response.")
            break

    # 4. Save to CSV
    csv_filename = os.path.join(
        "data", f"kp{int(round(Kp))}_freq{int(round(cli_args.freq))}hz.csv"
    )
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Time", "Target Motor Position", "Motor Position", "Motor Velocity", "Motor Applied Torque", "Motor calculated Torque", "Ankle Position", "Ankle Velocity", "Ankle Applied Torque", "Ankle Calculated Torque", "Kp", "Kd"])
        # Write data rows
        for t, target, actual, mvel, mt, mct, ap, av, at, act, kp, kd in zip(time_history, target_history, actual_pos_history, motor_vel_history, motor_torque_history, motor_desired_torque_history, ankle_pos_history, ankle_vel_history, ankle_torque_history, ankle_desired_torque_history, Kp_history, Kd_history):
            writer.writerow([t, target, actual, mvel, mt, mct, ap, av, at, act, kp, kd])

    print(f"Successfully saved data to {csv_filename} for posterior analysis.")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close the app cleanly so a subprocess-based sweep doesn't hang
        # waiting on a leftover Isaac Sim process.
        app.close()
