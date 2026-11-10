"""Throughput / physics benchmark for a trained ProtoMotions checkpoint (IsaacLab, headless).

Mirrors protomotions/inference_agent.py setup, then runs the deterministic policy for
--warmup + --steps control steps and reports env-steps/s. Optional:
  --profile : per-phase wall-clock breakdown (adds cuda syncs, so totals are slightly inflated)
  --trace   : save per-step traces (obs of first envs, skin forces, motor torque, errors) to .npz
  --no-policy : zero actions, no checkpoint weights (still needs a config dir, see --config-dir).
              Limited: worked for magic configs on smpl10bis_tissue; some reward terms assume
              policy-shaped actions and fail (for real throughput, time a short train_agent run).

Two runs with the same seed are bit-identical on GPU PhysX, so --trace before/after a refactor
checks that it does not change the simulation. BENCH_REPO=<path> runs another checkout (e.g. a
git worktree of the previous commit) for that comparison.

    python scripts/bench_sim.py --checkpoint results/<run>/last.ckpt --num-envs 16 --steps 150 --trace a.npz
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = os.environ.get("BENCH_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "protomotions"))
os.chdir(REPO)

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--config-dir", default=None, help="dir with resolved_configs_inference.pt + experiment_config.py")
p.add_argument("--motion-file", default=None)
p.add_argument("--num-envs", type=int, default=2048)
p.add_argument("--warmup", type=int, default=30)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--pos-iters", type=int, default=None)
p.add_argument("--vel-iters", type=int, default=None)
p.add_argument("--fps", type=int, default=None)
p.add_argument("--decimation", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--trace", default=None)
p.add_argument("--trace-envs", type=int, default=8)
p.add_argument("--profile", action="store_true")
p.add_argument("--label", default="")
p.add_argument("--results", default=None, help="jsonl file to append the summary to")
p.add_argument("--no-policy", action="store_true", help="zero actions, no agent (for robots without a checkpoint)")
args = p.parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def fill_missing_defaults(obj, seen=None):
    """Old pickled configs can miss fields added to the dataclasses later: set them to defaults."""
    import dataclasses

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name not in obj.__dict__:
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default_factory())
            fill_missing_defaults(obj.__dict__.get(f.name), seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            fill_missing_defaults(v, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            fill_missing_defaults(v, seen)


def main():
    seed_all(args.seed)
    checkpoint = Path(args.checkpoint)
    cfg_dir = Path(args.config_dir) if args.config_dir else checkpoint.parent
    rc = torch.load(cfg_dir / "resolved_configs_inference.pt", map_location="cpu", weights_only=False)
    fill_missing_defaults(rc)
    robot_config, simulator_config = rc["robot"], rc["simulator"]
    terrain_config, scene_lib_config = rc.get("terrain"), rc["scene_lib"]
    motion_lib_config, env_config, agent_config = rc["motion_lib"], rc["env"], rc["agent"]

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = True
    if args.motion_file:
        motion_lib_config.motion_file = args.motion_file
    if args.pos_iters is not None:
        simulator_config.sim.physx.num_position_iterations = args.pos_iters
    if args.vel_iters is not None:
        simulator_config.sim.physx.num_velocity_iterations = args.vel_iters
    if args.fps is not None:
        simulator_config.sim.fps = args.fps
    if args.decimation is not None:
        simulator_config.sim.decimation = args.decimation

    import importlib.util

    spec = importlib.util.spec_from_file_location("ckpt_exp", cfg_dir / "experiment_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_inference_overrides(robot_config, simulator_config, env_config, agent_config, args)

    fabric = Fabric(**FabricConfig(devices=1, num_nodes=1, loggers=[], callbacks=[]).to_dict())
    fabric.launch()
    app_launcher = AppLauncher({"headless": True, "device": str(fabric.device)})
    seed_all(args.seed)

    from protomotions.utils.component_builder import build_all_components

    comps = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        simulation_app=app_launcher.app,
    )
    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=comps["terrain"],
        scene_lib=comps["scene_lib"],
        motion_lib=comps["motion_lib"],
        simulator=comps["simulator"],
    )
    if args.no_policy:
        act_size = env.robot_config.number_of_actions + int(os.environ.get("BENCH_EXTRA_ACTIONS", "2"))

        def policy_action(obs):
            return torch.zeros(args.num_envs, act_size, device=fabric.device)

        # the simulator's action buffers start at DOF size; give them the full size up front
        env.simulator._common_actions = torch.zeros(args.num_envs, act_size, device=fabric.device)
        env.simulator._previous_actions = env.simulator._common_actions.clone()

    AgentClass = get_class(agent_config._target_)
    agent = None if args.no_policy else AgentClass(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
    if agent is not None:
        agent.setup()
    # The merged robot has fewer bodies, which changes the CRITIC inputs (max_coords_obs /
    # mimic_target_poses over all bodies). Rollouts only use the actors, so skip critic tensors
    # whose shape differs (bench only).
    from protomotions.agents.ppo.model import PPOModel

    _orig_load = PPOModel.load_state_dict

    def _load_skip_critic_mismatch(self, sd, strict=True, **kw):
        own = self.state_dict()
        bad = [k for k, v in sd.items() if k.startswith("_critic") and k in own and own[k].shape != v.shape]
        if bad:
            print(f"[bench] skipping {len(bad)} critic tensors with a new shape (actor loaded in full)")
            sd = {k: v for k, v in sd.items() if k not in bad}
            missing, unexpected = _orig_load(self, sd, strict=False, **kw)
            assert not [k for k in missing if not k.startswith("_critic")] and not unexpected, (missing, unexpected)
            return missing, unexpected
        return _orig_load(self, sd, strict=strict, **kw)

    PPOModel.load_state_dict = _load_skip_critic_mismatch
    if agent is not None:
        agent.load(str(checkpoint), load_env=False)
    ev = agent.evaluator if agent is not None else None
    h_agent = ev.agent if ev is not None else None
    p_agent = getattr(ev, "prosthetic_agent", None)
    if h_agent is not None:
        h_agent.eval()
    if p_agent is not None:
        p_agent.eval()

    def _policy_action(obs):
        obs = h_agent.add_agent_info_to_obs(obs)
        obs_td = h_agent.obs_dict_to_tensordict(obs)
        ho = h_agent.model(obs_td)
        a_h = ho["mean_action"] if "mean_action" in ho else ho["action"]
        if p_agent is None:
            return h_agent.expand_action_to_env(a_h)
        po = p_agent.model(obs_td)
        a_p = po["mean_action"] if "mean_action" in po else po["action"]
        return (h_agent.expand_action_to_env(a_h, num_extra_actions=2)
                + p_agent.expand_action_to_env(a_p, num_extra_actions=2))

    if not args.no_policy:
        policy_action = _policy_action

    sim = env.simulator
    print(f"[bench] physx pos/vel iters = {simulator_config.sim.physx.num_position_iterations}/"
          f"{simulator_config.sim.physx.num_velocity_iterations}, fps={simulator_config.sim.fps}, "
          f"decimation={simulator_config.sim.decimation}, num_bodies={sim._robot.num_bodies}")

    # ---------------- optional per-phase profiling ----------------
    prof = {}

    def timed(name, fn):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            prof[name] = prof.get(name, 0.0) + time.perf_counter() - t
            return out
        return wrapper

    if args.profile:
        sim._sim.step = timed("physx_step", sim._sim.step)
        sim._apply_control = timed("apply_control(+sensor_tap)", sim._apply_control)
        sim._scene.write_data_to_sim = timed("write_data_to_sim", sim._scene.write_data_to_sim)
        sim._scene.update = timed("scene_update", sim._scene.update)
        if sim.sensor_tap is not None:
            sim.sensor_tap = timed("sensor_tap", sim.sensor_tap)
        env.post_physics_step = timed("env_post_physics_step", env.post_physics_step)

    # ---------------- trace setup ----------------
    names = list(sim._robot.data.body_names)
    jnames = list(sim._robot.data.joint_names)
    skin_ids = [i for i, n in enumerate(names) if n.startswith("skin_box")]
    motor_j = jnames.index("Motor")  # noqa
    slide_j = jnames.index("suspension_slide")
    ankle_j = jnames.index("R_Ankle_y")
    tr = {k: [] for k in ["obs", "skin_f", "motor_tau", "slide", "ankle", "gt_err", "done", "term"]}
    TE = min(args.trace_envs, args.num_envs)

    def skin_forces():
        # [E, 8, 3] world-frame net contact force on each skin box (sim body order)
        return sim._get_simulator_bodies_contact_buf().rigid_body_contact_forces[:, skin_ids]

    done_indices = None
    n_total = args.warmup + args.steps
    n_resets = 0
    n_term = 0
    t0 = None
    with torch.no_grad():
        for step in range(n_total):
            if step == args.warmup:
                torch.cuda.synchronize()
                prof.clear()
                t0 = time.perf_counter()
                n_resets = 0
                n_term = 0
            obs, _ = env.reset(done_indices)
            obs, rewards, dones, terminated, extras = env.step(policy_action(obs))
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if step >= args.warmup:
                n_resets += done_indices.numel()
                n_term += int(terminated.sum().item()) if terminated is not None else 0
            if args.trace:
                o = obs if isinstance(obs, dict) else {"obs": obs}
                tr["obs"].append(torch.cat([v[:TE].reshape(TE, -1).float() for k, v in sorted(o.items())
                                            if torch.is_tensor(v)], dim=-1).cpu())
                tr["skin_f"].append(skin_forces().cpu())
                tr["motor_tau"].append(sim._robot.data.applied_torque[:, motor_j].cpu())
                tr["slide"].append(sim._robot.data.joint_pos[:, slide_j].cpu())
                tr["ankle"].append(sim._robot.data.joint_pos[:, ankle_j].cpu())
                ge = extras.get("mimic_other/gt_err")
                tr["gt_err"].append(ge.cpu() if ge is not None else torch.zeros(args.num_envs))
                tr["done"].append(dones.cpu())
                tr["term"].append(terminated.cpu() if terminated is not None else torch.zeros(args.num_envs))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.steps * args.num_envs / dt
    summary = dict(
        label=args.label, num_envs=args.num_envs, steps=args.steps,
        pos_iters=simulator_config.sim.physx.num_position_iterations,
        vel_iters=simulator_config.sim.physx.num_velocity_iterations,
        num_bodies=int(sim._robot.num_bodies),
        fps=simulator_config.sim.fps, decimation=simulator_config.sim.decimation,
        wall_s=round(dt, 3), ms_per_ctrl_step=round(1000 * dt / args.steps, 3),
        env_steps_per_s=round(sps, 1), resets=n_resets, terminations=n_term,
        profile=None if not args.profile else {k: round(1000 * v / args.steps, 3) for k, v in prof.items()},
    )
    print("[bench] RESULT " + json.dumps(summary))
    if args.results:
        with open(args.results, "a") as f:
            f.write(json.dumps(summary) + "\n")
    if args.trace:
        np.savez_compressed(args.trace, **{k: torch.stack(v).numpy() for k, v in tr.items()},
                            skin_names=np.array([names[i] for i in skin_ids]))
        print(f"[bench] trace saved to {args.trace}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip slow Kit shutdown


if __name__ == "__main__":
    main()
