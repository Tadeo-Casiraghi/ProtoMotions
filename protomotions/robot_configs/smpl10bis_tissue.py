# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""smpl10bis_merged with the residual-limb soft tissue modelled as joints ("hybrid" socket).

Assets: protomotions/data/assets/scripts/make_tissue_socket.py (from the merged assets).
Each of the 8 skin patches sits on a prismatic "<patch>_tissue" joint along its socket normal, with a
spring/damper drive; the patch touches the socket through a rigid, frictionless contact. Suspension
(liner, suction, sleeve) is the suspension_slide / _x / _y / _z drives. Joint drives are solved
exactly by PhysX, so the socket motion no longer depends on the solver iteration count the way the
compliant-contact version did.

Calibration (scripts/socket_bench.py --protocol calib, 1440 Hz / 128 iterations reference):
  * tissue stiffness = TISSUE_SCALE x the skin material stiffness of the smpl10bis asset, per region
    (posterior 46666, medial 74500, lateral 104000, anterior 140000 N/m: ratios 1 : 1.60 : 2.23 : 3.00);
    damping = TISSUE_SCALE x 2000 Ns/m for every region, as the skin materials had
  * suspension_slide: 27000 N/m, 1000 Ns/m, resting target +22 mm (socket side, + = away from the limb):
    unloaded, the suspension carries the prosthesis (~3 kg) with the tissue nearly unloaded
  * suspension_x / _y / _z: 350 Nm/rad, 175 Nms/rad (liner coupling; torsion is mostly this)
Measured: ~15 mm pistoning from unloaded to 800 N (a third of it in the first 160 N), 3.2 mm at a
90 N pull-out, ~0.3 deg of socket rotation per 60 Nm, ~1 deg per 10 Nm of torsion. Published static
values for transtibial sockets: ~10 mm loading -> unloading, ~4/6/8 mm at 30/60/90 N pull-out with a
pin lock and ~70 % less with a suction liner (Gholizadeh et al. 2011, 2012).
"""
from dataclasses import dataclass, field
from typing import Dict

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import ControlConfig, ControlType, RobotAssetConfig, SimulatorParams
from protomotions.robot_configs.smpl10bis_merged import Smpl10bisMergedRobotConfig
from protomotions.simulator.base_simulator.motor_model import MotorModelConfig
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabPhysXParams, IsaacLabSimParams
from protomotions.simulator.newton.config import NewtonSimParams

TISSUE_SCALE = 4.0
SKIN_REGION_STIFFNESS = {"posterior": 46666.0, "medial": 74500.0, "lateral": 104000.0, "anterior": 140000.0}
SKIN_DAMPING = 2000.0


def _tissue_control_info() -> Dict[str, ControlInfo]:
    return {
        f"skin_box_{region}_.*_tissue": ControlInfo(
            stiffness=TISSUE_SCALE * k,
            damping=TISSUE_SCALE * SKIN_DAMPING,
            armature=0.0,
            friction=0.0,
            effort_limit=1.0e5,
            velocity_limit=100.0,
        )
        for region, k in SKIN_REGION_STIFFNESS.items()
    }


def _control() -> ControlConfig:
    base = Smpl10bisMergedRobotConfig().control
    overrides = dict(base.override_control_info)
    overrides.update(
        {
            "suspension_slide": ControlInfo(
                stiffness=27000.0, damping=1000.0, armature=0.0, friction=0.0, effort_limit=5000.0, velocity_limit=20.0
            ),
            **{
                f"suspension_{a}": ControlInfo(
                    stiffness=350.0, damping=175.0, armature=0.0, friction=0.0, effort_limit=2000.0, velocity_limit=100.0
                )
                for a in ("x", "y", "z")
            },
            **_tissue_control_info(),
        }
    )
    return ControlConfig(
        control_type=ControlType.BUILT_IN_PD_HYBRID,
        torque_joints=list(base.torque_joints),
        motor_model=MotorModelConfig(),
        override_control_info=overrides,
    )


@dataclass
class Smpl10bisTissueRobotConfig(Smpl10bisMergedRobotConfig):
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/smpl_humanoid10bis_tissue.xml",
            usd_asset_file_name="usd/smpl_humanoid10bis_tissue.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
            max_linear_velocity=1000.0,
            max_angular_velocity=100.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(default_factory=_control)

    # 270 Hz physics, 90 Hz control. With the tissue joints the socket is within ~0.5 mm RMS of a
    # 1440 Hz / 128-iteration reference at 16 iterations (socket_bench.py), and training runs ~1.4x
    # faster than smpl10bis_merged at 360 Hz / 32. 180 Hz is faster still but the motor model's
    # 3 ms torque lag is shorter than its 5.6 ms step, which changes the actuator dynamics.
    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(fps=60, decimation=2, substeps=2),
            isaaclab=IsaacLabSimParams(
                fps=270,
                decimation=3,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=16,
                    num_velocity_iterations=1,
                    contact_offset=0.002,
                    max_depenetration_velocity=3,
                ),
            ),
            genesis=GenesisSimParams(fps=60, decimation=2, substeps=2),
            newton=NewtonSimParams(fps=120, decimation=4),
        )
    )

    # Position targets of passive joints (read by the experiment's passive_defaults_by_name)
    passive_dof_targets: Dict[str, float] = field(default_factory=lambda: {"suspension_slide": 0.022})
