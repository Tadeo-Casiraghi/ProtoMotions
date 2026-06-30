import argparse
import csv
import os
import sys
import time
from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Parse sweep parameters BEFORE launching the app.
#
# --freq            physics simulation frequency in Hz (physics_dt = 1/freq)
# --pos-iters       PhysX num_position_iterations
# --vel-iters       PhysX num_velocity_iterations
# --contact-offset / --max-depen  exposed for completeness but held fixed by
#                   the sweep runner -- only dt/pos_iters/vel_iters are varied
# --save-movement   write the per-step angle/pistoning trace to data2/
#                   (off by default so pure timing runs don't produce
#                   duplicate movement CSVs)
# --interactive     launch with a GUI viewport instead of headless (for manual
#                   single-run debugging, not used by the sweep runner)
# --plot            save matplotlib PNGs of the recorded traces to
#                   data2/plots/ (saved to disk rather than plt.show(), since
#                   the point here is unattended automation)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=float, default=480.0, help="Physics sim frequency in Hz")
parser.add_argument("--pos-iters", type=int, default=16, help="PhysX num_position_iterations")
parser.add_argument("--vel-iters", type=int, default=8, help="PhysX num_velocity_iterations")
parser.add_argument("--contact-offset", type=float, default=0.002)
parser.add_argument("--max-depen", type=float, default=1.0)
parser.add_argument("--duration", type=float, default=0.9, help="Total simulated seconds to run")
parser.add_argument("--save-movement", action="store_true", help="Save the per-step trace CSV to data2/")
parser.add_argument("--interactive", action="store_true", help="Launch with a GUI viewport instead of headless")
parser.add_argument("--plot", action="store_true", help="Save PNG plots of the traces to data2/plots/")
parser.add_argument("--custom-name", type=str, default="", help="Optional custom name to output files for this run")
cli_args = parser.parse_args()

physics_dt = 1.0 / cli_args.freq

# Tag used for every artifact from this run so movement CSVs / plots / log
# lines are easy to match up to a setting.
RUN_TAG = f"freq{int(round(cli_args.freq))}hz_pos{cli_args.pos_iters}_vel{cli_args.vel_iters}"

if cli_args.custom_name:
    RUN_TAG = cli_args.custom_name

# 1. Start the Isaac Sim application
launcher = AppLauncher(headless=not cli_args.interactive)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg, GroundPlaneCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import numpy as np


@configclass
class ProstheticTestSceneCfg(InteractiveSceneCfg):

    prosthetic_leg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ProstheticLeg",
        spawn=sim_utils.UsdFileCfg(
            usd_path="prosthetic_leg_test.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=cli_args.pos_iters,
                solver_velocity_iteration_count=cli_args.vel_iters,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=False,  # free-falling
                solver_position_iteration_count=cli_args.pos_iters,
                solver_velocity_iteration_count=cli_args.vel_iters,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "suspension_slide": ImplicitActuatorCfg(
                joint_names_expr=["suspension_slide.*"],
                stiffness=1000.0, # 20000.0,
                damping=0.0, # 0.1,
                armature=0.0,
                effort_limit=10000.0,
                velocity_limit=10000.0,
                friction=20.0,
            ),
            "suspension_y": ImplicitActuatorCfg(
                joint_names_expr=["suspension_y.*"],
                stiffness=10.0,
                damping=1.0,
                armature=0.0,
                effort_limit=10000.0,
                velocity_limit=10000.0,
                friction=0.0,
            ),
            "suspension_x": ImplicitActuatorCfg(
                joint_names_expr=["suspension_x.*"],
                stiffness=10.0,
                damping=1.0,
                armature=0.0,
                effort_limit=10000.0,
                velocity_limit=10000.0,
                friction=0.0,
            ),
            "suspension_z": ImplicitActuatorCfg(
                joint_names_expr=["suspension_z.*"],
                stiffness=10.0,
                damping=1.0,
                armature=0.0,
                effort_limit=10000.0,
                velocity_limit=10000.0,
                friction=0.0,
            ),
        }
    )


def main():
    if cli_args.save_movement or cli_args.plot:
        os.makedirs("data2", exist_ok=True)

    scene_cfg = ProstheticTestSceneCfg(num_envs=1, env_spacing=2.0)

    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(visible=True),
    )

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt)

    # Create PhysX config
    sim_cfg.physx = sim_utils.PhysxCfg()
    # sim_cfg.physx.num_position_iterations = cli_args.pos_iters
    # sim_cfg.physx.num_velocity_iterations = cli_args.vel_iters
    # sim_cfg.physx.contact_offset = cli_args.contact_offset
    # sim_cfg.physx.rest_offset = -0.01
    sim_cfg.physx.max_depenetration_velocity = cli_args.max_depen
    sim_cfg.physx.enable_external_forces_every_iteration=True

    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot: Articulation = scene["prosthetic_leg"]

    joint_idx = robot.find_joints("suspension_slide")[0]
    joint_idx0 = robot.find_joints("suspension_x")[0]
    joint_idx1 = robot.find_joints("suspension_y")[0]
    joint_idx2 = robot.find_joints("suspension_z")[0]

    # Data collection lists
    time_history = []
    pos_history = []
    angle0_history = []
    angle1_history = []
    angle2_history = []

    current_time = 0.0
    target_positions = torch.zeros((1, robot.num_joints), device=robot.device)

    print(f"[{RUN_TAG}] physics_dt={physics_dt:.6f}s  pos_iters={cli_args.pos_iters}  "
          f"vel_iters={cli_args.vel_iters}  contact_offset={cli_args.contact_offset}  "
          f"max_depen={cli_args.max_depen}  duration={cli_args.duration}s")

    # Time only the stepping loop -- this (not app/asset startup) is what we
    # actually want to compare across dt / iteration-count settings.
    loop_start = time.time()

    while launcher.app.is_running():
        target_positions[:, joint_idx] = -0.15
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

        time_history.append(current_time)
        pos_history.append(current_pos)
        angle0_history.append(current_angle0)
        angle1_history.append(current_angle1)
        angle2_history.append(current_angle2)

        if current_time > cli_args.duration:
            break

    loop_elapsed = time.time() - loop_start
    print("Finished recording fall response.")

    if cli_args.save_movement:
        csv_filename = os.path.join("data2", f"movement_{RUN_TAG}.csv")
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Pistoning (suspension_slide)", "Angle_X", "Angle_Y", "Angle_Z"])
            for t, p, a0, a1, a2 in zip(time_history, pos_history, angle0_history, angle1_history, angle2_history):
                writer.writerow([t, p, a0, a1, a2])
        print(f"Saved movement data to {csv_filename}")

    # Machine-readable line for the sweep runner to parse out of stdout.
    print(f"SIM_WALL_TIME_SEC: {loop_elapsed:.6f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close()
        # Isaac Sim can leave a non-daemon background thread alive after
        # close(), which hangs a normal interpreter exit. Force-terminate so
        # the sweep runner waiting on this process doesn't get stuck.
        os._exit(exit_code)
