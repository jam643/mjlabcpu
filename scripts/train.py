"""Generic PPO trainer — trains any registered env.

Usage:
    uv run python scripts/train.py cartpole
    uv run python scripts/train.py panda_push --timesteps 1_000_000 --envs 8

For live viewer during training (macOS requires mjpython):
    mjpython scripts/train.py cartpole --render

Available envs: any module in examples/envs/ that exposes make_env().
Point at a different registry with --envs-dir.
Per-env PPO defaults come from the env module's ppo_cfg() if defined.
All PPO hyperparameters can be overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

from mjlabcpu.training import PPOCfg, PPOTrainer

# Default env registry — override with --envs-dir
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_ENVS_DIR = _PROJECT_ROOT / "examples" / "envs"


def load_env_module(name: str, envs_dir: pathlib.Path):
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


def main() -> None:
    # --- Pre-parse env name to load per-env defaults before building full parser ---
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("env", nargs="?", default=None)
    pre.add_argument("--envs", type=int, default=None)
    pre.add_argument("--envs-dir", type=pathlib.Path, default=_DEFAULT_ENVS_DIR)
    known, _ = pre.parse_known_args()

    envs_dir = known.envs_dir
    num_envs_hint = known.envs or 4
    base_cfg = PPOCfg(num_envs=num_envs_hint)  # fallback defaults

    if known.env:
        mod = load_env_module(known.env, envs_dir)
        if hasattr(mod, "ppo_cfg"):
            base_cfg = mod.ppo_cfg(num_envs_hint)

    # --- Full parser with per-env defaults ---
    parser = argparse.ArgumentParser(
        description="Train any mjlabcpu environment with PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env", help="Env name (matches <envs-dir>/<name>.py)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=base_cfg.num_envs)
    parser.add_argument("--render", action="store_true", help="Open passive viewer on env 0")
    parser.add_argument(
        "--save", type=str, default=None,
        help="Checkpoint path (default: checkpoints/<env>_ppo.pkl)",
    )
    parser.add_argument(
        "--wandb", type=str, default=None, metavar="PROJECT",
        help="W&B project name",
    )
    parser.add_argument(
        "--envs-dir", type=pathlib.Path, default=_DEFAULT_ENVS_DIR,
        help=f"Directory containing env modules (default: {_DEFAULT_ENVS_DIR})",
    )
    # PPO hyperparameter overrides
    parser.add_argument("--lr", type=float, default=base_cfg.learning_rate)
    parser.add_argument("--num-steps", type=int, default=base_cfg.num_steps)
    parser.add_argument("--gamma", type=float, default=base_cfg.gamma)
    parser.add_argument("--clip-coef", type=float, default=base_cfg.clip_coef)
    args = parser.parse_args()

    save_path = args.save or f"checkpoints/{args.env}_ppo.pkl"
    render_mode = "human" if args.render else None

    # Reload module (may have been loaded with wrong num_envs hint above)
    mod = load_env_module(args.env, args.envs_dir)

    print(f"Building env '{args.env}': {args.envs} envs, render_mode={render_mode!r}")
    env = mod.make_env(num_envs=args.envs, render_mode=render_mode)
    print(f"  obs={env.observation_space.shape[0]}  act={env.action_space.shape[0]}")

    # Merge per-env defaults with any CLI overrides
    base = mod.ppo_cfg(args.envs) if hasattr(mod, "ppo_cfg") else PPOCfg(num_envs=args.envs)
    cfg = PPOCfg(
        num_steps=args.num_steps,
        num_envs=args.envs,
        learning_rate=args.lr,
        num_epochs=base.num_epochs,
        num_minibatches=base.num_minibatches,
        gamma=args.gamma,
        gae_lambda=base.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=base.ent_coef,
        vf_coef=base.vf_coef,
        max_grad_norm=base.max_grad_norm,
        hidden_sizes=base.hidden_sizes,
        log_interval=base.log_interval,
        wandb_project=args.wandb,
    )

    trainer = PPOTrainer(env, cfg)
    print(f"Training for {args.timesteps:,} timesteps...")
    metrics = trainer.train(total_timesteps=args.timesteps)

    final = metrics["mean_reward"][-1] if metrics["mean_reward"] else float("nan")
    print(f"\nTraining complete. Final mean reward: {final:.3f}")

    trainer.save(save_path)
    print(f"Saved checkpoint: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
