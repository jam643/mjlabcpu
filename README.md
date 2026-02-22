# mjlabcpu

Isaac Lab-style manager-based RL environment API running on **CPU MuJoCo + JAX JIT**. Define environments declaratively with observation, reward, termination, action, and event managers — then train with the built-in pure-JAX PPO trainer or any standard gymnasium-compatible algorithm.

## Design

Two backends are available — CPU MuJoCo (always available) and MJX (JAX-native, GPU-ready):

```
CPU backend                          MJX backend
───────────────────────────────────  ────────────────────────────────────────
MuJoCo (C, CPU)  ←→  JAX JIT        MuJoCo MJX (JAX) — fully fused JIT
    mj_step()     numpy bridge           vmap(mjx.step) × decimation
    parallel envs obs/reward/done        obs / reward / done / reset
                  PPOTrainer             ↓ one JAX→numpy boundary per step
                  gymnasium.Env          gymnasium.Env
```

**CPU backend**: Physics runs on CPU MuJoCo (no GPU required). Observations, rewards, and terminations are pure JAX, compiled once with `jax.jit`. State crosses the MuJoCo→JAX boundary as numpy arrays each step. ~6,500 env-steps/sec on 4 cartpole envs (M-series Mac).

**MJX backend**: The entire env step — physics, observations, rewards, terminations, and conditional reset — is a single `@jax.jit` kernel. No Python-side loop over envs; `jax.vmap` batches physics and `jax.lax.scan` unrolls decimation. ~21,000 env-steps/sec on 64 cartpole envs (M-series Mac CPU), 50k+ on GPU.

## Installation

```bash
git clone <repo>
cd mjlabcpu
uv sync
uv pip install -e .
```

Requires Python ≥ 3.10. Core dependencies: `mujoco>=3.2.0`, `jax[cpu]>=0.4.20`, `flax>=0.8.0`, `optax>=0.2.0`, `gymnasium>=0.29.0`.

Optional extras:

```bash
uv pip install mjlabcpu[mjx]    # MuJoCo MJX backend (GPU-ready)
uv pip install mjlabcpu[train]  # Weights & Biases logging
uv pip install mjlabcpu[viz]    # rerun live monitoring dashboard
uv pip install "imageio[ffmpeg]"  # save RGB videos from --rgb mode
```

## Quick Start

### View any environment

`scripts/view.py` works with any registered env. It supports interactive, headless, live-plot, and trained-policy playback modes.

```bash
# Interactive MuJoCo viewer (macOS — requires mjpython)
mjpython scripts/view.py cartpole
mjpython scripts/view.py panda_push

# Headless RGB video (works with standard python)
uv run python scripts/view.py cartpole --rgb --steps 500
uv run python scripts/view.py panda_push --rgb --out panda.mp4

# Random policy (default) or zero action
uv run python scripts/view.py cartpole --policy zero

# Play back a trained checkpoint (deterministic policy)
mjpython scripts/view.py cartpole --checkpoint checkpoints/cartpole.pkl
uv run python scripts/view.py cartpole --rgb --checkpoint checkpoints/cartpole.pkl --episodes 5 --out cartpole_trained.mp4
```

> **macOS note:** The interactive viewer requires `mjpython`. Offscreen `--rgb` mode works with regular `python`.

`--checkpoint` auto-restores the network architecture from the saved config and runs deterministic inference. Per-episode rewards are printed whenever a checkpoint is loaded or `--episodes` is set.

### Live monitoring dashboard

Pass `--plot` to open a [rerun](https://rerun.io) dashboard alongside the viewer. It streams reward terms, observation terms (per dimension), actions (grouped by term), and termination signals in real time. The `step_in_episode` timeline resets at each episode boundary so the live view always shows the current episode; old episodes remain accessible via the time scrubber.

```bash
# Interactive viewer + live dashboard
mjpython scripts/view.py cartpole --plot
mjpython scripts/view.py panda_push --plot

# Headless mode also works
uv run python scripts/view.py cartpole --rgb --plot
```

The rerun viewer opens automatically as a native process — no conflict with the MuJoCo passive viewer.

### Train with PPO

`scripts/train.py` trains any registered env. Per-env PPO defaults come from the env module's `ppo_cfg()` if defined; all hyperparameters can be overridden via CLI flags.

```bash
# Cartpole
uv run python scripts/train.py cartpole --timesteps 500_000 --envs 4

# Panda push
uv run python scripts/train.py panda_push --timesteps 2_000_000 --envs 8

# Log to Weights & Biases
uv run python scripts/train.py cartpole --wandb mjlabcpu

# Open MuJoCo viewer during training (macOS: mjpython)
mjpython scripts/train.py cartpole --render

# Save checkpoint
uv run python scripts/train.py cartpole --save checkpoints/cartpole.pkl
```

## Environments

Envs live in `examples/envs/` as plain Python modules exposing `make_env(num_envs, render_mode)` and optionally `ppo_cfg(num_envs)`. Point `scripts/view.py` or `scripts/train.py` at a different directory with `--envs-dir`.

### cartpole

Cart-pole balance. 4-DOF system; goal is to keep the pole upright.

| | |
|---|---|
| Obs | joint positions + velocities (4) |
| Act | cart position target (1) |
| Rewards | `is_alive` + `cartpole_upright` (×2) + `action_rate_l2` (×−0.01) |
| Episode | 10 s / 1250 steps; terminates on pole angle > 0.2 rad or cart pos > 2.4 m |

### panda_push

Franka Panda arm pushes a puck to a randomised goal position on a flat surface. Requires asset download:

```bash
uv run python examples/download_panda_assets.py
```

| | |
|---|---|
| Obs | joint pos + vel (14), EEF XY (2), puck pos (3), goal pos (3) = 22 total |
| Act | 7-DOF joint position targets (scaled ×0.3 around defaults) |
| Rewards | `object_to_goal` (×5) + `eef_to_object` (×1) + `action_rate_l2` (×−0.01) + `joint_vel_l2` (×−0.001) |
| Episode | 8 s / 1000 steps; terminates on puck > 1.5 m from origin |
| Commands | `goal_pos` — uniform XY goal resampled each episode |

## Defining an Environment

Environments are configured entirely through dataclasses — no subclassing required.

```python
from mjlabcpu.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlabcpu.envs.mdp import observations as obs_mdp, rewards as rew_mdp
from mjlabcpu.envs.mdp import terminations as term_mdp, events as event_mdp
from mjlabcpu.envs.mdp.actions import JointPositionAction
from mjlabcpu.managers import (
    ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, SceneEntityCfg, TerminationTermCfg,
)
from mjlabcpu.scene import SceneCfg
from mjlabcpu.sim import SimulationCfg
from mjlabcpu.entity import EntityCfg

entity_cfg = SceneEntityCfg(name="cartpole")

cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
        num_envs=4,
        entities={"cartpole": EntityCfg(prim_path="cartpole", spawn="assets/cartpole.xml")},
    ),
    sim=SimulationCfg(dt=0.002),
    episode_length_s=10.0,
    decimation=4,  # 4 physics steps per RL step → dt_rl = 0.008s

    observations={
        "policy": ObservationGroupCfg(terms={
            "joint_pos": ObservationTermCfg(func=obs_mdp.joint_pos_rel, params={"entity_cfg": entity_cfg}),
            "joint_vel": ObservationTermCfg(func=obs_mdp.joint_vel_rel, params={"entity_cfg": entity_cfg}),
        })
    },
    rewards={
        "alive":          RewardTermCfg(func=rew_mdp.is_alive, weight=1.0),
        "upright":        RewardTermCfg(func=rew_mdp.cartpole_upright, params={"entity_cfg": entity_cfg}, weight=2.0),
        "action_penalty": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01),
    },
    terminations={
        "time_out": TerminationTermCfg(func=term_mdp.time_out, params={"max_episode_length": 1250}, time_out=True),
        "fallen":   TerminationTermCfg(func=term_mdp.cartpole_fallen, params={"entity_cfg": entity_cfg, "max_pole_angle": 0.2}, time_out=False),
    },
    actions={
        "cart_drive": ActionTermCfg(cls=JointPositionAction, params={"entity_cfg": entity_cfg, "scale": 1.0}),
    },
    events={
        "reset": EventTermCfg(func=event_mdp.reset_joints_uniform, mode="reset",
                              params={"entity_name": "cartpole", "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}),
    },
)

env = ManagerBasedRlEnv(cfg)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Training with PPO

```python
from mjlabcpu.training import PPOTrainer, PPOCfg

trainer = PPOTrainer(env, PPOCfg(
    num_steps=512,
    num_envs=4,
    learning_rate=3e-4,
    num_epochs=10,
    num_minibatches=4,
    hidden_sizes=(256, 128),
    wandb_project="mjlabcpu",  # optional — omit to disable
))

metrics = trainer.train(total_timesteps=500_000)
trainer.save("checkpoints/cartpole_ppo.pkl")
```

The network forward pass and PPO update step are both JIT-compiled. `metrics` is a dict of loss/reward lists for plotting.

To run inference after training:

```python
trainer.load("checkpoints/cartpole_ppo.pkl")
actions = trainer.get_action(obs, deterministic=True)  # (num_envs, act_dim)
```

### Weights & Biases

When `wandb_project` is set, metrics are streamed live to [wandb.ai](https://wandb.ai) every `log_interval` updates:

| Key | Description |
|---|---|
| `train/reward_mean` | Mean episode reward across envs |
| `train/pg_loss` | Policy gradient loss |
| `train/value_loss` | Value function loss |
| `train/entropy` | Policy entropy |
| `train/steps_per_sec` | Training throughput |

## Live Monitoring

`EnvMonitor` logs per-term reward, observation, and action data to rerun without touching the JIT hot path.

```python
from mjlabcpu.utils import EnvMonitor

monitor = EnvMonitor(env)           # spawns rerun viewer
obs_terms = env._obs_manager.compute_terms(env._make_dummy_state())
monitor.log_step(obs_terms, rewards, terminated, truncated, info, action)
```

Two timelines are maintained: `step_in_episode` (resets to 0 each episode — used for the live view) and `total_step` (monotonically increasing — for cross-episode context).

## MJX Backend

Use `MjxManagerBasedRlEnv` instead of `ManagerBasedRlEnv` to run with the MJX backend. The config API is identical:

```python
from mjlabcpu.envs.mjx_env import MjxManagerBasedRlEnv

env = MjxManagerBasedRlEnv(cfg)  # same cfg as CPU backend
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(actions)
```

The entire env step — physics (×decimation via `lax.scan`), observations, rewards, terminations, and conditional reset with per-env randomisation — is fused into a single `@jax.jit` kernel. There is only one JAX→numpy boundary per `step()` call.

**Limitations vs CPU backend:**
- Only `reset_joints_uniform` events are supported inside the fused kernel (other event functions are skipped)
- Only `"rgb_array"` render mode is available (no passive viewer for MJX)
- Requires `mujoco-mjx` to be installed (`pip install mujoco-mjx`)

## Rendering

```python
# Offscreen RGB (works with standard python)
env = ManagerBasedRlEnv(cfg, render_mode="rgb_array")
env.reset()
frame = env.render()  # np.ndarray (480, 640, 3) uint8

# Interactive viewer on env 0 (macOS: use mjpython)
env = ManagerBasedRlEnv(cfg, render_mode="human")
env.reset()
env.render()  # opens passive viewer window
```

## Built-in MDP Functions

### Observations (`mjlabcpu.envs.mdp.observations`)

| Function | Description | Shape |
|---|---|---|
| `joint_pos_rel` | Joint positions relative to defaults | `(N, nq)` |
| `joint_vel_rel` | Joint velocities | `(N, nv)` |
| `base_lin_vel` | Root linear velocity in body frame | `(N, 3)` |
| `base_ang_vel` | Root angular velocity in body frame | `(N, 3)` |
| `projected_gravity` | Gravity vector in body frame | `(N, 3)` |
| `root_pos_w` | Root position in world frame | `(N, 3)` |
| `root_quat_w` | Root quaternion (wxyz) in world frame | `(N, 4)` |
| `body_pos_w_xy` | Named body XY position in world frame | `(N, 2)` |
| `last_action` | Previous action | `(N, act_dim)` |
| `generated_commands` | Named command tensor | `(N, cmd_dim)` |

### Rewards (`mjlabcpu.envs.mdp.rewards`)

| Function | Description |
|---|---|
| `is_alive` | +1 per step |
| `action_rate_l2` | Penalise action change (smoothness) |
| `joint_vel_l2` | Penalise joint velocities |
| `joint_torques_l2` | Penalise control effort |
| `flat_orientation_l2` | Penalise base tilt |
| `upright` | Reward upright root orientation |
| `cartpole_upright` | Reward cartpole pole angle |
| `track_lin_vel_xy` | Gaussian XY velocity tracking |
| `track_ang_vel_z` | Gaussian yaw rate tracking |
| `object_to_goal` | Negative distance from object to goal command |
| `eef_to_object` | Negative distance from end-effector to object |

### Terminations (`mjlabcpu.envs.mdp.terminations`)

| Function | Description |
|---|---|
| `time_out` | Truncate after `max_episode_length` steps |
| `cartpole_fallen` | Terminate on pole angle or cart position limit |
| `object_out_of_bounds` | Terminate when object exceeds XY radius |

### Events (`mjlabcpu.envs.mdp.events`)

| Function | Mode | Description |
|---|---|---|
| `reset_joints_uniform` | `reset` | Randomise joint positions and velocities on reset |
| `reset_root_state_uniform` | `reset` | Randomise root position/orientation on reset |

## Writing Custom MDP Functions

All observation, reward, and termination functions share the same signature:

```python
# Observation function
def my_obs(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    return state.qpos[:, entity_cfg.qpos_addrs]  # (num_envs, nq)

# Reward function
def my_reward(state: SimState) -> jnp.ndarray:
    return jnp.ones(state.qpos.shape[0])  # (num_envs,)

# Termination function  — return bool array
def my_termination(state: SimState, threshold: float) -> jnp.ndarray:
    return state.qpos[:, 0] > threshold  # (num_envs,) bool
```

Functions must be pure JAX (no Python control flow on traced values). Extra parameters beyond `state` are resolved at manager init time via `functools.partial`.

## Project Structure

```
src/mjlabcpu/
├── sim/
│   ├── sim.py           # Simulation — wraps N mjData, runs parallel mj_step
│   ├── sim_state.py     # SimState — JAX pytree snapshot of MuJoCo state
│   └── mjx_sim.py       # MjxSimulation — batched mjx.Data JAX pytree
├── scene/
│   └── scene.py         # Scene — assembles MjSpec from EntityCfg list
├── entity/
│   ├── entity.py        # Entity — name/index registry
│   └── data.py          # EntityData — per-entity JAX array views
├── managers/
│   ├── action_manager.py       # Writes actions to mjData ctrl (+ compute_ctrl_jax for MJX)
│   ├── observation_manager.py  # JIT-compiled obs computation + compute_terms() for monitoring
│   ├── reward_manager.py       # JIT-compiled reward computation
│   ├── termination_manager.py  # JIT-compiled termination logic
│   ├── event_manager.py        # Reset event dispatch
│   └── command_manager.py      # Command resampling
├── envs/
│   ├── manager_based_rl_env.py  # CPU gymnasium.Env (render_mode support)
│   ├── mjx_env.py               # MJX gymnasium.Env (fully fused JIT step)
│   └── mdp/                     # Built-in observation/reward/termination/action/event functions
├── training/
│   ├── ppo.py           # PPOTrainer + PPOCfg
│   ├── networks.py      # ActorCritic Flax network
│   └── rollout.py       # RolloutBuffer + GAE computation
└── utils/
    ├── math.py          # JAX quaternion helpers (wxyz convention)
    └── monitor.py       # EnvMonitor — live rerun dashboard (optional rerun-sdk)
scripts/
├── view.py              # Generic viewer: --rgb, --plot, --checkpoint, --episodes, --policy
└── train.py             # Generic PPO trainer: --timesteps, --envs, --wandb, --render
examples/
├── envs/
│   ├── cartpole.py      # Cart-pole balance env (make_env + ppo_cfg)
│   └── panda_push.py    # Franka Panda push env (make_env + ppo_cfg)
├── assets/
│   ├── cartpole.xml
│   └── panda_push/      # Downloaded by download_panda_assets.py
└── download_panda_assets.py
tests/                   # pytest suite (49 tests)
```

## Running Tests

```bash
uv run pytest
# Skip MJX-specific tests if mujoco-mjx is not installed
uv run pytest --ignore=tests/test_mjx.py
```

## Related Projects

- **[MjLab](https://github.com/mujocolab/mjlab)** — the closest sibling project: same Isaac Lab manager-based API concept, but targets the MuJoCo Warp backend (GPU via CUDA). `mjlabcpu` targets CPU MuJoCo and MJX (JAX-native, GPU-ready via XLA) instead. The two projects were developed independently.
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — the original manager-based RL framework this API is modelled after, built on NVIDIA Omniverse/Isaac Sim.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.
