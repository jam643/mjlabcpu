"""Generic environment viewer — runs any registered env.

Interactive viewer (macOS requires mjpython):
    mjpython scripts/view.py cartpole
    mjpython scripts/view.py panda_push

    # Load a trained checkpoint (deterministic policy)
    mjpython scripts/view.py cartpole --checkpoint checkpoints/cartpole.pkl
    mjpython scripts/view.py cartpole --checkpoint checkpoints/cartpole.pkl --episodes 10

    # Manual control via viewer actuator sliders + live reward plots
    mjpython scripts/view.py panda_push --manual --plot

Headless RGB video (works with regular python):
    uv run python scripts/view.py cartpole --rgb --steps 500
    uv run python scripts/view.py panda_push --rgb --out panda.mp4
    uv run python scripts/view.py cartpole --rgb --checkpoint checkpoints/cartpole.pkl --episodes 5

Live monitoring dashboard (requires rerun-sdk):
    mjpython scripts/view.py cartpole --plot
    uv run python scripts/view.py cartpole --rgb --plot

Available envs: any module in examples/envs/ that exposes make_env().
Point at a different registry with --envs-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from collections.abc import Callable

import numpy as np

# Default env registry — override with --envs-dir
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_ENVS_DIR = _PROJECT_ROOT / "examples" / "envs"


def load_env_module(name: str, envs_dir: pathlib.Path):
    import importlib.util

    path = envs_dir / f"{name}.py"
    if not path.exists():
        available = [p.stem for p in envs_dir.glob("*.py") if p.stem != "__init__"]
        raise SystemExit(
            f"Unknown env '{name}'. Available: {', '.join(sorted(available))}\n"
            f"Add {envs_dir}/{name}.py with a make_env() function to register it."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_action_fn(env, policy: str, checkpoint: str | None) -> tuple[Callable, bool]:
    """Build an action callable from a checkpoint path or policy name.

    Args:
        env:        Constructed gymnasium env (used to read obs/act dims).
        policy:     ``"random"`` or ``"zero"`` — used when no checkpoint.
        checkpoint: Path to a ``.pkl`` PPO checkpoint, or ``None``.

    Returns:
        action_fn:       ``obs -> actions`` callable.
        print_episodes:  Whether to print per-episode reward summaries.
    """
    if checkpoint is not None:
        import dataclasses
        import pickle

        from mjlabcpu.training import PPOCfg, PPOTrainer

        # Peek at checkpoint to restore the exact network architecture.
        with open(checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        cfg = ckpt.get("cfg", PPOCfg())
        # Override num_envs to match viewer (always 1 env for visualisation).
        cfg = dataclasses.replace(cfg, num_envs=env.num_envs)

        trainer = PPOTrainer(env, cfg)
        trainer.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}  (hidden={cfg.hidden_sizes})")

        def action_fn(obs: np.ndarray) -> np.ndarray:
            return trainer.get_action(obs, deterministic=True)

        return action_fn, True

    act_dim = env.action_space.shape[0]
    if policy == "zero":
        return (
            lambda obs: np.zeros((env.num_envs, act_dim), dtype=np.float32),
            False,
        )
    # random
    return (
        lambda obs: env.action_space.sample()[None].astype(np.float32),
        False,
    )


def run_human(
    env_name: str,
    envs_dir: pathlib.Path,
    policy: str,
    checkpoint: str | None,
    plot: bool = False,
    max_episodes: int | None = None,
) -> None:
    """Open a passive MuJoCo viewer and loop until it is closed (or max_episodes)."""
    from mjlabcpu.utils import EnvMonitor

    mod = load_env_module(env_name, envs_dir)
    env = mod.make_env(num_envs=1, render_mode="human")
    obs, _ = env.reset()
    env.render()  # open viewer window

    action_fn, print_episodes = _make_action_fn(env, policy, checkpoint)
    target_dt = env.dt  # seconds per RL step

    print(
        f"Env: {env_name}  |  obs={env.observation_space.shape[0]}"
        f"  act={env.action_space.shape[0]}"
        f"  |  step_dt={target_dt * 1000:.1f} ms  |  close viewer to quit"
    )
    if max_episodes:
        print(f"Will stop after {max_episodes} episode(s).")

    monitor = EnvMonitor(env) if plot else None

    step = 0
    episode = 0
    ep_acc = 0.0
    ep_rewards: list[float] = []

    while env.is_viewer_running():
        t0 = time.perf_counter()

        action = action_fn(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

        ep_acc += float(rewards[0])

        if monitor:
            obs_terms = env._obs_manager.compute_terms(env._make_dummy_state())
            monitor.log_step(obs_terms, rewards, terminated, truncated, info, action)

        if terminated[0] or truncated[0]:
            episode += 1
            ep_rewards.append(ep_acc)
            if print_episodes or max_episodes:
                print(f"  episode {episode:3d}  reward={ep_acc:+.1f}")
            ep_acc = 0.0
            obs, _ = env.reset()
            if max_episodes and episode >= max_episodes:
                break

        step += 1

        # Throttle to real-time
        elapsed = time.perf_counter() - t0
        sleep_for = target_dt - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    env.close()
    if ep_rewards:
        print(f"\n{episode} episode(s)  mean reward={np.mean(ep_rewards):+.1f}")
    print(f"Viewer closed after {step} steps.")


def run_human_manual(
    env_name: str,
    envs_dir: pathlib.Path,
    plot: bool = False,
    max_episodes: int | None = None,
) -> None:
    """Open a passive MuJoCo viewer with actuator sliders for manual control.

    The RL action pipeline is bypassed entirely — viewer sliders write directly
    to ``mjData.ctrl`` and physics uses those values.  Rewards and observations
    are still computed each step and (if ``--plot``) streamed to rerun so you
    can see exactly what the reward functions see as you manually move the arm.
    """
    from mjlabcpu.utils import EnvMonitor

    mod = load_env_module(env_name, envs_dir)
    env = mod.make_env(num_envs=1, render_mode="human")
    env.reset()
    env.render()  # open viewer window

    # Placeholder action array for monitor logging (sliders bypass RL actions)
    action_zeros = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)
    monitor = EnvMonitor(env) if plot else None
    target_dt = env.dt

    print(
        f"MANUAL CONTROL — use the viewer actuator sliders to drive the robot.\n"
        f"Env: {env_name}  |  obs={env.observation_space.shape[0]}"
        f"  act={env.action_space.shape[0]}"
        f"  |  step_dt={target_dt * 1000:.1f} ms  |  close viewer to quit"
    )
    if max_episodes:
        print(f"Will stop after {max_episodes} episode(s).")

    step = 0
    episode = 0
    ep_acc = 0.0

    while env.is_viewer_running():
        t0 = time.perf_counter()

        # --- Physics step without overriding ctrl ---
        # Viewer sliders write to data[0].ctrl directly; we step physics
        # without calling apply_actions() so those values are preserved.
        env._episode_length = env._episode_length + 1
        for _ in range(env.cfg.decimation):
            env._sim.step()
        env._command_manager.step(target_dt)
        env.render()

        # --- Compute state / rewards / terminations (same as normal step) ---
        state = env._make_dummy_state()
        total_reward, reward_terms = env._reward_manager.compute(state)
        done, truncated, term_terms = env._termination_manager.compute(state)
        terminated = done & ~truncated

        rewards_np = np.array(total_reward)
        terminated_np = np.array(terminated)
        truncated_np = np.array(truncated)
        info = {
            "reward_terms": {k: np.array(v) for k, v in reward_terms.items()},
            "termination_terms": {k: np.array(v) for k, v in term_terms.items()},
        }

        ep_acc += float(rewards_np[0])

        if monitor:
            obs_terms = env._obs_manager.compute_terms(state)
            monitor.log_step(obs_terms, rewards_np, terminated_np, truncated_np, info, action_zeros)

        # --- Episode reset ---
        done_ids = np.where(np.array(done))[0].tolist()
        if done_ids:
            episode += 1
            print(f"  episode {episode:3d}  reward={ep_acc:+.1f}")
            ep_acc = 0.0
            env._reset_envs(done_ids)
            if max_episodes and episode >= max_episodes:
                break

        step += 1

        # Throttle to real-time
        elapsed = time.perf_counter() - t0
        sleep_for = target_dt - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    env.close()
    if episode:
        print(f"\n{episode} episode(s) completed.")
    print(f"Viewer closed after {step} steps.")


def run_rgb(
    env_name: str,
    envs_dir: pathlib.Path,
    policy: str,
    checkpoint: str | None,
    num_steps: int,
    out_path: str,
    plot: bool = False,
    max_episodes: int | None = None,
    photo: bool = False,
) -> None:
    """Render to RGB frames and save as video."""
    from mjlabcpu.utils import EnvMonitor

    mod = load_env_module(env_name, envs_dir)
    env = mod.make_env(num_envs=1, render_mode="rgb_array")
    obs, _ = env.reset()

    action_fn, print_episodes = _make_action_fn(env, policy, checkpoint)

    frames: list[np.ndarray] = []
    monitor = EnvMonitor(env) if plot else None

    # --- Optional photorealistic renderer ---
    photo_renderer = None
    if photo:
        try:
            from mjlabcpu.render import PhotoRenderer
        except ImportError as exc:
            print(f"PhotoRenderer import failed: {exc}")
            import sys

            sys.exit(1)
        print("[PhotoRenderer] Initializing Blender Cycles scene (one-time cost)…")
        try:
            photo_renderer = PhotoRenderer(env._sim.model, width=640, height=480)
        except FileNotFoundError:
            import sys

            print(
                "Blender not found. Install it first:\n"
                "  brew install --cask blender\n"
                "or download from https://www.blender.org/download/ and move to /Applications."
            )
            sys.exit(1)
        print("[PhotoRenderer] Ready.")

    episode = 0
    ep_acc = 0.0
    ep_rewards: list[float] = []

    limit = f"{num_steps} steps" + (f" / {max_episodes} episodes" if max_episodes else "")
    renderer_label = "Blender Cycles" if photo else "MuJoCo"
    print(f"Rendering '{env_name}' ({limit}) via {renderer_label} → {out_path}")

    t0 = time.perf_counter()
    for _ in range(num_steps):
        action = action_fn(obs)
        obs, rewards, terminated, truncated, info = env.step(action)

        if photo_renderer is not None:
            photo_renderer.update(env._sim.data[0])
            frames.append(photo_renderer.render())
        else:
            frames.append(env.render())

        ep_acc += float(rewards[0])

        if monitor:
            obs_terms = env._obs_manager.compute_terms(env._make_dummy_state())
            monitor.log_step(obs_terms, rewards, terminated, truncated, info, action)

        if terminated[0] or truncated[0]:
            episode += 1
            ep_rewards.append(ep_acc)
            if print_episodes or max_episodes:
                print(f"  episode {episode:3d}  reward={ep_acc:+.1f}")
            ep_acc = 0.0
            obs, _ = env.reset()
            if max_episodes and episode >= max_episodes:
                break

    elapsed = time.perf_counter() - t0
    if photo_renderer is not None:
        photo_renderer.close()
    env.close()
    print(f"Rendered {len(frames)} frames in {elapsed:.1f}s  ({len(frames) / elapsed:.1f} fps)")
    if ep_rewards:
        print(f"{episode} episode(s)  mean reward={np.mean(ep_rewards):+.1f}")

    _save_video(frames, out_path)


def _save_video(frames: list[np.ndarray], path: str) -> None:
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=30)
        print(f"Saved: {path}")
        return
    except ImportError:
        pass
    try:
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved: {path}")
        return
    except ImportError:
        pass
    npy = path.rsplit(".", 1)[0] + ".npy"
    np.save(npy, np.stack(frames))
    print(f"imageio/opencv not found — saved frames to {npy}")
    print("Install imageio:  uv pip install 'imageio[ffmpeg]'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize any mjlabcpu environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("env", help="Env name (matches <envs-dir>/<name>.py)")

    # Policy
    policy_group = parser.add_argument_group("policy")
    policy_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a trained PPO checkpoint (.pkl). Runs deterministic policy.",
    )
    policy_group.add_argument(
        "--policy",
        choices=["random", "zero"],
        default="random",
        help="Fallback policy when no --checkpoint is given (default: random).",
    )

    # Episode control
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N complete episodes (default: run until viewer closes / steps exhausted).",
    )

    # Rendering
    render_group = parser.add_argument_group("rendering")
    render_group.add_argument(
        "--rgb",
        action="store_true",
        help="Render to video file instead of interactive viewer.",
    )
    render_group.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Steps to render in --rgb mode (default: 500).",
    )
    render_group.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output video path for --rgb mode (default: <env>.mp4).",
    )
    render_group.add_argument(
        "--photo",
        action="store_true",
        help="Use photorealistic Blender Cycles renderer (requires bpy; --rgb mode only).",
    )

    # Extras
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Manual control mode: bypass the RL policy entirely and use the "
            "viewer's built-in actuator sliders to drive the robot. "
            "Rewards are still computed and streamed to rerun (combine with --plot). "
            "Requires interactive viewer; incompatible with --rgb."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open rerun live-plot dashboard.",
    )
    parser.add_argument(
        "--envs-dir",
        type=pathlib.Path,
        default=_DEFAULT_ENVS_DIR,
        help=f"Directory containing env modules (default: {_DEFAULT_ENVS_DIR}).",
    )

    args = parser.parse_args()

    if args.checkpoint is not None and not pathlib.Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    if args.rgb:
        if args.manual:
            print("Note: --manual requires interactive viewer; ignoring --manual in --rgb mode.")
        out = args.out or f"{args.env}.mp4"
        run_rgb(
            args.env,
            args.envs_dir,
            args.policy,
            args.checkpoint,
            args.steps,
            out,
            plot=args.plot,
            max_episodes=args.episodes,
            photo=args.photo,
        )
    elif args.manual:
        if args.photo:
            print("Note: --photo requires --rgb. Ignoring --photo flag.")
        run_human_manual(
            args.env,
            args.envs_dir,
            plot=args.plot,
            max_episodes=args.episodes,
        )
    else:
        if args.photo:
            print("Note: --photo requires --rgb. Ignoring --photo flag.")

        run_human(
            args.env,
            args.envs_dir,
            args.policy,
            args.checkpoint,
            plot=args.plot,
            max_episodes=args.episodes,
        )


if __name__ == "__main__":
    main()
