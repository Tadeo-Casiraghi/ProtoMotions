import argparse
import csv
from isaaclab.app import AppLauncher

# 1. Start the Isaac Sim application
launcher = AppLauncher(headless=False)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


@configclass
class ProstheticTestSceneCfg(InteractiveSceneCfg):
    
    prosthetic_leg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ProstheticLeg",
        spawn=sim_utils.UsdFileCfg(
            usd_path="prosthetic_leg_test.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True, # Fixed in the air
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), 
            joint_pos={".*": 0.0}, 
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Joints that arent 
            "suspension_slide": ImplicitActuatorCfg(
                joint_names_expr=["suspension_slide.*"], 
                stiffness=2000.0,         # Adjust to tune response
                damping=2000.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=100000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=1000000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
            "suspension_y": ImplicitActuatorCfg(
                joint_names_expr=["suspension_y.*"],
                stiffness=5000.0,         # Adjust to tune response
                damping=500.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=10000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
            "suspension_x": ImplicitActuatorCfg(
                joint_names_expr=["suspension_x.*"],
                stiffness=5000.0,         # Adjust to tune response
                damping=500.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=10000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
            "suspension_z": ImplicitActuatorCfg(
                joint_names_expr=["suspension_z.*"],
                stiffness=5000.0,         # Adjust to tune response
                damping=500.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=10000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
            "R_Ankle_y": ImplicitActuatorCfg(
                joint_names_expr=["R_Ankle_y.*"],
                stiffness=500.0,         # Adjust to tune response
                damping=10.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=10000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
            "Motor": ImplicitActuatorCfg(
                joint_names_expr=["Motor.*"],
                stiffness=50.0,         # Adjust to tune response
                damping=0.01,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=1000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
                friction=0.0,              # No friction for clean step response data
            ),
        }
    )

def main():
    scene_cfg = ProstheticTestSceneCfg(num_envs=1, env_spacing=2.0)
    decimation = 4
    
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)

    # Create PhysX config
    sim_cfg.physx = sim_utils.PhysxCfg()

    # Now set fields AFTER creation
    sim_cfg.physx.num_position_iterations = 16
    sim_cfg.physx.num_velocity_iterations = 8
    sim_cfg.physx.contact_offset = 0.002
    sim_cfg.physx.max_depenetration_velocity = 1.0

    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    
    robot: Articulation = scene["prosthetic_leg"]
    
    # Identify joint to test
    test_joint_name = "Motor" 
    joint_idx = robot.find_joints(test_joint_name)[0]
    
    # Data collection lists
    time_history = []
    target_history = []
    actual_pos_history = []
    
    current_time = 0.0
    
    # Step Disturbance Parameters
    initial_target = 0.0
    step_target = 1.0      # New target position (radians)
    disturbance_applied = False
    time_since_disturbance = 0.0
    post_disturbance_duration = 5.0 # Record for 5 seconds AFTER the step is applied
    
    velocity_threshold = 0.1 # Threshold to consider the leg "stable"
    
    print("Initializing Simulation... Waiting for leg to stabilize.")
    target_positions = torch.zeros((1, robot.num_joints), device=robot.device)
    
    step_count = 0
    while launcher.app.is_running():
        current_vel = robot.data.joint_vel[0, joint_idx].item()
        current_pos = robot.data.joint_pos[0, joint_idx].item()

        print("Position:", current_pos, "   Velocity:", current_vel)
        
        if step_count % decimation == 0:
            # 1. Check for stability to apply the step disturbance
            if not disturbance_applied:
                # If velocity is near zero AND we've simulated for at least 2.0s to let initial drops settle
                if abs(current_vel) < velocity_threshold and current_time > 2.0:
                    print(f"Leg stabilized at t={current_time:.2f}s. Applying Step Disturbance!")
                    disturbance_applied = True
                    current_target = step_target
                else:
                    current_target = initial_target
            else:
                current_target = step_target
                time_since_disturbance += sim_cfg.dt * decimation
            
            target_positions[:, joint_idx] = current_target
            robot.set_joint_position_target(target_positions)
        
        scene.write_data_to_sim()

        # 2. Step physics
        sim.step()
        scene.update(sim.get_physics_dt())
        
        # 3. Record Data (only start recording right before the disturbance for clean data, or record all)
        time_history.append(current_time)
        target_history.append(current_target)
        actual_pos_history.append(current_pos)
        
        current_time += sim_cfg.dt
        print(current_time)
        step_count += 1

        # End simulation after recording the post-disturbance response
        if disturbance_applied and time_since_disturbance >= post_disturbance_duration:
            print("Finished recording step response.")
            break

    # 4. Save to CSV
    csv_filename = "step_response_data.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["Time (s)", "Target Position (rad)", "Actual Position (rad)"])
        # Write data rows
        for t, target, actual in zip(time_history, target_history, actual_pos_history):
            writer.writerow([t, target, actual])
            
    print(f"Successfully saved data to {csv_filename} for posterior analysis.")

if __name__ == "__main__":
    main()
