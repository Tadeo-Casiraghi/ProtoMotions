"""Socket <-> residual-limb test bench on the real training robot.

The pelvis is fixed, the human joints are held by their PD drives, the suspension and tissue joints
use the robot config drives, and known loads are applied to the prosthesis (osl_ankle_assembly),
one load case per env. Records at 90 Hz: socket DOFs (suspension_slide, suspension_x/y/z), the
tissue joints (if the robot has them) and per-skin-patch contact force magnitudes, to an .npz.

--protocol steps (default): load steps
    0 compress   : +800 N along +z (stance, foot pushed up into the socket)
    1 sag_moment : +800 N and 60 Nm about world x
    2 front_mom  : +800 N and 60 Nm about world y
    3 torsion    : +800 N and 10 Nm about world z (limb long axis)
    4 tension    : -150 N (swing, socket pulled off the limb)
    5 shear      : +800 N and 150 N along +x
    (s): 0-1 settle | 1-1.2 ramp up | 1.2-1.7 hold | 1.7-1.9 ramp down | 1.9-2.5 free | 2.5-4.5 1 Hz cycles

--protocol calib: quasi-static ramps (0 -> 800 N compression, 0 -> 90 N pull-out), 60 Nm / 10 Nm
    moments on a 400 N standing load, and a 1 Hz gait-like vertical GRF (see calib_loads)

--protocol stand: ground plane, pelvis fixed at 12 heights around default_root_height
    (STAND_GRID="start,step" in m), vertical load on each foot: standing leg-length parity

Examples:
    python scripts/socket_bench.py --robot smpl10bis_tissue --protocol calib --fps 360 --pos-iters 16 \\
        --slide-target 0.022 --out calib.npz
    python scripts/socket_bench.py --robot smpl10bis_tissue --protocol stand --slide-target 0.022 --out stand.npz
Compare a setting against a high-rate reference (e.g. --fps 1440 --pos-iters 128) to measure solver error.
The --skin-joints / --extend-plates / --filter-bone options build the tissue model at runtime on
the merged asset (used to prototype smpl10bis_tissue before its assets existed).
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

p = argparse.ArgumentParser()
p.add_argument("--robot", default="smpl10bis_merged")
p.add_argument("--fps", type=int, default=360)
p.add_argument("--pos-iters", type=int, default=32)
p.add_argument("--vel-iters", type=int, default=1)
p.add_argument("--solver", type=int, default=1, help="0 PGS, 1 TGS")
p.add_argument("--out", required=True)
p.add_argument("--duration", type=float, default=4.5)
p.add_argument("--load-scale", type=float, default=1.0)
p.add_argument("--protocol", default="steps", choices=["steps", "calib", "stand"],
               help="steps: load steps per case; calib: slow ramps + standing moments + 1 Hz gait GRF")
# experiment knobs (defaults = training model)
p.add_argument("--skin-k-scale", type=float, default=1.0, help="scale every skin compliant stiffness")
p.add_argument("--skin-d", type=float, default=None, help="skin compliant damping (default: asset value)")
p.add_argument("--skin-friction", type=float, default=None, help="override skin static/dynamic friction")
p.add_argument("--susp-armature", type=float, default=None, help="armature on suspension joints")
p.add_argument("--slide-k", type=float, default=None)
p.add_argument("--slide-d", type=float, default=None)
p.add_argument("--slide-target", type=float, default=-0.075)
p.add_argument("--rot-k", type=float, default=None)
p.add_argument("--rot-d", type=float, default=None)
p.add_argument("--no-skin-contact", action="store_true", help="disable skin box collisions (joint-only model)")
p.add_argument("--contact-offset", type=float, default=0.002)
p.add_argument("--max-depen", type=float, default=3.0)
# hybrid model: each skin box on a prismatic joint (local z = socket normal) with a drive, rigid contact
p.add_argument("--skin-joints", action="store_true")
p.add_argument("--skin-joint-alpha", type=float, default=0.8, help="box drive stiffness = alpha * region stiffness")
p.add_argument("--skin-joint-d", type=float, default=8000.0, help="box drive damping (Ns/m), same for every region")
p.add_argument("--skin-joint-range", type=float, nargs=2, default=(-0.002, 0.025), help="tissue travel limits (m)")
p.add_argument("--filter-bone", action="store_true", help="no R_Knee (bone) <-> socket_assembly collision")
p.add_argument("--debug-socket-contacts", type=float, default=None, help="print socket_assembly contact partners at this time (s)")
p.add_argument("--extend-plates", type=float, default=0.0, help="extend socket plates along the limb axis by this much at each end (m)")
p.add_argument("--tors-k", type=float, default=None, help="suspension_y (long axis) drive stiffness")
p.add_argument("--tors-d", type=float, default=None)
p.add_argument("--bend-k", type=float, default=None, help="suspension_x/z drive stiffness")
p.add_argument("--bend-d", type=float, default=None)
args = p.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import PhysxCfg  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402

from protomotions.robot_configs.factory import robot_config as make_robot_config  # noqa: E402

SKIN = [f"skin_box_{s}_{l}" for l in ("top", "bottom") for s in ("posterior", "medial", "anterior", "lateral")]
SUSP = ["suspension_slide", "suspension_x", "suspension_y", "suspension_z"]
CASES = ["compress", "sag_moment", "front_mom", "torsion", "tension", "shear"]
CALIB_CASES = ["comp_ramp", "tens_ramp", "stand_sag", "stand_front", "stand_tors", "gait"]
import os as _os
_s0, _ds = (float(v) for v in _os.environ.get("STAND_GRID", "-0.040,0.005").split(","))
STAND_DH = [_s0 + _ds * i for i in range(12)]  # pelvis height offsets around default_root_height
if args.protocol == "stand":
    CASES = [f"dh{dh*1000:+.0f}mm" for dh in STAND_DH]
    if args.duration == 4.5:
        args.duration = 2.5
if args.protocol == "calib":
    CASES = CALIB_CASES
    if args.duration == 4.5:
        args.duration = 4.0


def ramp(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def calib_loads(t, L):
    """Per-env (force, torque) in world frame on the prosthesis, for --protocol calib.
    0 comp_ramp  : 0 -> 800 N compression over 1-3 s (quasi-static), hold
    1 tens_ramp  : 0 -> 90 N pull-out over 1-3 s, hold
    2-4 stand_*  : 400 N standing load (1-1.3 s), then 0 -> 60 Nm (sag/front) or 0 -> 10 Nm (torsion) over 1.5-3 s
    5 gait       : from 1 s, 1 Hz cycles: 0.62 s stance with a double-hump vertical GRF (peak ~1.1 x 800 N), swing 0 N
    """
    F = np.zeros((6, 3)); T = np.zeros((6, 3))
    F[0, 2] = 800 * ramp(t, 1.0, 3.0)
    F[1, 2] = -90 * ramp(t, 1.0, 3.0)
    stand = 400 * ramp(t, 1.0, 1.3)
    F[2:5, 2] = stand
    m = ramp(t, 1.5, 3.0)
    T[2, 0] = 60 * m; T[3, 1] = 60 * m; T[4, 2] = 10 * m
    if t >= 1.0:
        tau = ((t - 1.0) % 1.0) / 0.62
        if tau < 1.0:
            F[5, 2] = 800 * (1.15 * np.sin(np.pi * tau) + 0.3 * np.sin(3 * np.pi * tau))
    return F * L, T * L
NUM_ENVS = len(CASES)


def actuators_from(rc):
    """Same grouping as SceneCfg (one group per set of configured parameters)."""
    groups = {}
    for name, ci in rc.control.control_info.items():
        prm = {
            "stiffness": ci.stiffness, "damping": ci.damping, "armature": ci.armature,
            "effort_limit_sim": ci.effort_limit, "velocity_limit_sim": ci.velocity_limit, "friction": ci.friction,
        }
        if name == "Motor":  # torque-driven in training: hold it here so the prosthesis is rigid
            prm.update(stiffness=2000.0, damping=50.0, effort_limit_sim=2000.0)
        if name == "suspension_slide":
            if args.slide_k is not None:
                prm["stiffness"] = args.slide_k
            if args.slide_d is not None:
                prm["damping"] = args.slide_d
        if name in ("suspension_x", "suspension_y", "suspension_z"):
            if args.rot_k is not None:
                prm["stiffness"] = args.rot_k
            if args.rot_d is not None:
                prm["damping"] = args.rot_d
        if name == "suspension_y":
            if args.tors_k is not None:
                prm["stiffness"] = args.tors_k
            if args.tors_d is not None:
                prm["damping"] = args.tors_d
        if name in ("suspension_x", "suspension_z"):
            if args.bend_k is not None:
                prm["stiffness"] = args.bend_k
            if args.bend_d is not None:
                prm["damping"] = args.bend_d
        if name in SUSP and args.susp_armature is not None:
            prm["armature"] = args.susp_armature
        prm = {k: v for k, v in prm.items() if v is not None}
        g = groups.setdefault(tuple(prm), {"joints": [], "params": {}})
        g["joints"].append(name)
        for k, v in prm.items():
            g["params"].setdefault(k, {})[name] = v
    acts = {f"g{i}": ImplicitActuatorCfg(joint_names_expr=g["joints"], **g["params"]) for i, g in enumerate(groups.values())}
    if args.skin_joints:
        acts["skin"] = ImplicitActuatorCfg(
            joint_names_expr=[f"{b}_tissue" for b in SKIN],
            stiffness={f"{b}_tissue": args.skin_joint_alpha * REGION_K[b.split("_")[2]] for b in SKIN},
            damping=args.skin_joint_d, armature=0.0, effort_limit_sim=1e6, velocity_limit_sim=1e3, friction=0.0,
        )
    return acts


# region stiffness of the skin materials in the asset (N/m), read so the ratios are the asset's
REGION_K = {"posterior": 46666.0, "medial": 74500.0, "anterior": 140000.0, "lateral": 104000.0}


def add_skin_joints(stage, env):
    """Replace R_Knee -> skin box fixed joints by prismatic joints along the box local z (socket normal)."""
    from pxr import Gf, Sdf, UsdPhysics as P

    root = f"/World/envs/env_{env}/Robot"
    for b in SKIN:
        old = stage.GetPrimAtPath(f"{root}/joints/{b}")
        fj = P.Joint(old)
        b0, b1 = fj.GetBody0Rel().GetTargets(), fj.GetBody1Rel().GetTargets()
        lp0, lr0 = fj.GetLocalPos0Attr().Get(), fj.GetLocalRot0Attr().Get()
        lp1, lr1 = fj.GetLocalPos1Attr().Get(), fj.GetLocalRot1Attr().Get()
        assert lp1 == Gf.Vec3f(0, 0, 0) and lr1 == Gf.Quatf(1, 0, 0, 0), b  # joint frame == skin box frame
        old.SetActive(False)
        j = P.PrismaticJoint.Define(stage, f"{root}/joints/{b}_tissue")
        j.CreateBody0Rel().SetTargets(b0)
        j.CreateBody1Rel().SetTargets(b1)
        j.CreateLocalPos0Attr(lp0); j.CreateLocalRot0Attr(lr0)
        j.CreateLocalPos1Attr(lp1); j.CreateLocalRot1Attr(lr1)
        j.CreateAxisAttr("Z")
        j.CreateLowerLimitAttr(args.skin_joint_range[0]); j.CreateUpperLimitAttr(args.skin_joint_range[1])
        d = P.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force"); d.CreateTargetPositionAttr(0.0)
        d.CreateStiffnessAttr(1.0); d.CreateDampingAttr(1.0); d.CreateMaxForceAttr(1e6)


def main():
    rc = make_robot_config(args.robot)
    usd = os.path.join(REPO, "protomotions/data/assets", rc.asset.usd_asset_file_name)

    from isaaclab.utils import configclass

    @configclass
    class BenchScene(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                    max_linear_velocity=1000.0, max_angular_velocity=100.0, max_depenetration_velocity=args.max_depen,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, fix_root_link=True,
                    solver_position_iteration_count=args.pos_iters,
                    solver_velocity_iteration_count=args.vel_iters,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=args.contact_offset, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.4), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
            actuators=actuators_from(rc),
        )
        skin_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/bodies/(" + "|".join(SKIN) + ")")
        if args.protocol == "stand":
            feet_contacts = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/bodies/(L_Ankle|L_Toe|R_Ankle_foot|R_Ankle_toes)")

        if args.debug_socket_contacts is not None:
            socket_contacts = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/bodies/socket_assembly",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/bodies/.*"],
            )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device="cuda:0", dt=1.0 / args.fps,
        physx=PhysxCfg(solver_type=args.solver, max_position_iteration_count=max(args.pos_iters, 4),
                       max_velocity_iteration_count=max(args.vel_iters, 1), bounce_threshold_velocity=0.2,
                       gpu_max_rigid_patch_count=300000),
    ))
    scene_cfg = BenchScene(num_envs=NUM_ENVS, env_spacing=3.0)
    if args.protocol == "stand":
        from isaaclab.assets import AssetBaseCfg
        scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene = InteractiveScene(scene_cfg)

    # ---- optional material / collision edits on the spawned stage (before physics starts) ----
    stage = sim.stage
    for e in range(NUM_ENVS):
        if args.skin_joints:
            add_skin_joints(stage, e)
        if args.extend_plates > 0:
            from pxr import UsdGeom, Gf
            sa = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot/bodies/socket_assembly")
            for child in sa.GetChildren():
                if child.GetName().startswith("socket_box_") and child.GetName().endswith("_collisions"):
                    for g in child.GetChildren():  # the _geom_N Xform carrying the box scale
                        sc = g.GetAttribute("xformOp:scale")
                        v = sc.Get()
                        sc.Set(Gf.Vec3d(v[0], v[1] + args.extend_plates, v[2]))
        if args.filter_bone:
            from pxr import UsdPhysics as P
            knee = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot/bodies/R_Knee")
            P.FilteredPairsAPI.Apply(knee).CreateFilteredPairsRel().AddTarget(f"/World/envs/env_{e}/Robot/bodies/socket_assembly")
        looks = f"/World/envs/env_{e}/Robot/Looks"
        for m in ("Anterior", "Lateral", "Posterior", "Medial"):
            prim = stage.GetPrimAtPath(f"{looks}/{m}")
            if not prim:
                continue
            if args.skin_k_scale != 1.0:
                a = prim.GetAttribute("physxMaterial:compliantContactStiffness")
                a.Set(a.Get() * args.skin_k_scale)
            if args.skin_d is not None:
                prim.GetAttribute("physxMaterial:compliantContactDamping").Set(args.skin_d)
            if args.skin_friction is not None:
                prim.GetAttribute("physics:staticFriction").Set(args.skin_friction)
                prim.GetAttribute("physics:dynamicFriction").Set(args.skin_friction)
        if args.no_skin_contact:
            for b in SKIN:
                for prim in stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot/bodies/{b}").GetAllChildren():
                    for q in [prim] + list(prim.GetAllChildren()) + [c for ch in prim.GetAllChildren() for c in ch.GetAllChildren()]:
                        if q.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Set(False)

    sim.reset()
    robot: Articulation = scene["robot"]
    sensor = scene["skin_contacts"]
    dev = robot.device
    jn = list(robot.joint_names)
    susp_ids = [jn.index(n) for n in SUSP]
    tissue_ids = [jn.index(f"{b}_tissue") for b in SKIN] if f"{SKIN[0]}_tissue" in jn else []
    load_body = robot.find_bodies("osl_ankle_assembly")[0][0]
    sens_order = [list(sensor.body_names).index(n) for n in SKIN]

    if args.protocol == "stand":
        pose = robot.data.default_root_state[:, :7].clone()
        pose[:, :3] += scene.env_origins
        pose[:, 2] = rc.default_root_height + torch.tensor(STAND_DH, device=dev)
        robot.write_root_pose_to_sim(pose)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        feet = scene["feet_contacts"]
        foot_names = list(feet.body_names)
        left_ids = [foot_names.index(n) for n in ("L_Ankle", "L_Toe")]
        right_ids = [foot_names.index(n) for n in ("R_Ankle_foot", "R_Ankle_toes")]
        rec_lr = []
    target = torch.zeros(NUM_ENVS, robot.num_joints, device=dev)
    target[:, jn.index("suspension_slide")] = args.slide_target
    robot.set_joint_position_target(target)

    L = args.load_scale
    F = torch.zeros(NUM_ENVS, 1, 3, device=dev)
    T = torch.zeros(NUM_ENVS, 1, 3, device=dev)
    base_F = torch.tensor([[0, 0, 800], [0, 0, 800], [0, 0, 800], [0, 0, 800], [0, 0, -150], [150, 0, 800]],
                          dtype=torch.float, device=dev) * L
    base_T = torch.tensor([[0, 0, 0], [60, 0, 0], [0, 60, 0], [0, 0, 10], [0, 0, 0], [0, 0, 0]],
                          dtype=torch.float, device=dev) * L

    def profile(t):
        if t < 1.0:
            return 0.0
        if t < 1.2:
            return (t - 1.0) / 0.2
        if t < 1.7:
            return 1.0
        if t < 1.9:
            return 1.0 - (t - 1.7) / 0.2
        if t < 2.5:
            return 0.0
        return 0.5 - 0.5 * np.cos(2 * np.pi * (t - 2.5))  # 1 Hz, 0 -> 1 -> 0

    n_steps = int(round(args.duration * args.fps))
    every = args.fps // 90
    assert args.fps % 90 == 0, "fps must be a multiple of 90 (90 Hz recording grid)"
    rec_t, rec_q, rec_f, rec_tis = [], [], [], []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(n_steps):
        t = k / args.fps
        if args.protocol == "stand":
            F.zero_(); T.zero_()
        elif args.protocol == "calib":
            f_np, t_np = calib_loads(t, L)
            F[:, 0] = torch.as_tensor(f_np, dtype=torch.float, device=dev)
            T[:, 0] = torch.as_tensor(t_np, dtype=torch.float, device=dev)
        else:
            s = profile(t)
            F[:, 0] = base_F * s
            T[:, 0] = base_T * s
        robot.set_external_force_and_torque(F, T, body_ids=[load_body], is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1.0 / args.fps)
        if args.debug_socket_contacts is not None and abs(t - args.debug_socket_contacts) < 0.5 / args.fps:
            sc = scene["socket_contacts"]
            fm = sc.data.force_matrix_w[:, 0]  # [envs, filters, 3]
            names = sc.contact_physx_view.filter_patterns if hasattr(sc.contact_physx_view, "filter_patterns") else None
            body_names = list(robot.body_names)
            print(f"[debug] t={t:.3f} socket_assembly contact partners (|F| > 1 N):")
            for e in range(NUM_ENVS):
                mags = torch.linalg.norm(fm[e], dim=-1).cpu().numpy()
                partners = [(body_names[i] if len(mags) == len(body_names) else str(i), round(float(v), 1)) for i, v in enumerate(mags) if v > 1]
                print(f"   env {e} {CASES[e]:12s}: {partners}  slide {robot.data.joint_pos[e, jn.index('suspension_slide')].item()*1000:.1f} mm")
        if (k + 1) % every == 0:
            rec_t.append((k + 1) / args.fps)
            rec_q.append(robot.data.joint_pos[:, susp_ids].clone())
            rec_f.append(torch.linalg.norm(sensor.data.net_forces_w[:, sens_order], dim=-1).clone())
            if tissue_ids:
                rec_tis.append(robot.data.joint_pos[:, tissue_ids].clone())
            if args.protocol == "stand":
                fz = feet.data.net_forces_w[..., 2]
                rec_lr.append(torch.stack([fz[:, left_ids].sum(-1), fz[:, right_ids].sum(-1)], -1).clone())
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    np.savez_compressed(
        args.out, t=np.array(rec_t), q=torch.stack(rec_q).cpu().numpy(), f=torch.stack(rec_f).cpu().numpy(),
        tissue=torch.stack(rec_tis).cpu().numpy() if rec_tis else np.zeros(0), cases=np.array(CASES),
        lr=torch.stack(rec_lr).cpu().numpy() if args.protocol == "stand" else np.zeros(0),
        weight=float(robot.data.default_mass.sum() / NUM_ENVS * 9.81), dh=np.array(STAND_DH), susp=np.array(SUSP), skin=np.array(SKIN), wall=wall, args=str(vars(args)),
    )
    print(f"[socket_bench] {args.fps} Hz {args.pos_iters}/{args.vel_iters}: {n_steps} steps in {wall:.1f}s -> {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
