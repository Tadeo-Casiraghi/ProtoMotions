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
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import GroundPlaneCfg

import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


@configclass
class ProstheticTestSceneCfg(InteractiveSceneCfg):
    
    prosthetic_leg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ProstheticLeg",
        spawn=sim_utils.UsdFileCfg(
            usd_path="prosthetic_leg_test2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=False, # Fixed in the air
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5), 
            joint_pos={".*": 0.0}, 
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Joints that arent 
            "suspension_slide": ImplicitActuatorCfg(
                joint_names_expr=["suspension_slide.*"], 
                stiffness=20000.0,         # Adjust to tune response
                damping=500.0,            # Adjust to tune response
                armature=0.0,              # Adjust to tune response
                effort_limit=10000.0,     # Large effort limit to ensure it can apply the disturbance
                velocity_limit=10000.0,  # Large velocity limit to ensure it can apply the disturbance
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
        }
    )

def main():
    scene_cfg = ProstheticTestSceneCfg(num_envs=1, env_spacing=2.0)

    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(
            visible=True
        ),
    )

    decimation = 4
    
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / (120.0*4))

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
    
    # Data collection lists
    time_history = []
    pos_history = []
    angle0_history = []
    angle1_history = []
    angle2_history = []

    current_time = 0.0
    
    step_count = 0

    joint_idx = robot.find_joints("suspension_slide")[0]
    joint_idx0 = robot.find_joints("suspension_x")[0]
    joint_idx1 = robot.find_joints("suspension_y")[0]
    joint_idx2 = robot.find_joints("suspension_z")[0]

    target_positions = torch.zeros((1, robot.num_joints), device=robot.device)
    while launcher.app.is_running():
        target_positions[:, joint_idx] = -0.05
        robot.set_joint_position_target(target_positions)
        
        scene.write_data_to_sim()

        # 2. Step physics
        sim.step()
        scene.update(sim.get_physics_dt())

        current_pos = robot.data.joint_pos[0, joint_idx].item()
        current_angle0 = robot.data.joint_pos[0, joint_idx0].item()
        current_angle1 = robot.data.joint_pos[0, joint_idx1].item()
        current_angle2 = robot.data.joint_pos[0, joint_idx2].item()
        
        current_time += sim_cfg.dt
        step_count += 1

        time_history.append(current_time)
        pos_history.append(current_pos)
        angle0_history.append(current_angle0)
        angle1_history.append(current_angle1)
        angle2_history.append(current_angle2)

        # End simulation after recording the post-disturbance response
        if current_time > 0.8:  # Run for 10 seconds total
            print("Finished recording step response.")
            break

    # # 4. Save to CSV
    # csv_filename = "step_response_data.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write headers
    #     writer.writerow(["Time (s)", "Target Position (rad)", "Actual Position (rad)"])
    #     # Write data rows
    #     for t, target, actual in zip(time_history, target_history, actual_pos_history):
    #         writer.writerow([t, target, actual])
            
    # print(f"Successfully saved data to {csv_filename} for posterior analysis.")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time_history, pos_history)

    plt.figure()
    plt.plot(time_history, angle0_history)
    plt.plot(time_history, angle1_history)
    plt.plot(time_history, angle2_history)
    plt.show()


if __name__ == "__main__":
    main()
