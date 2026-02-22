# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repo provides an **Isaac Lab manager-based RL API** with a MuJoCo CPU/MJX backend. It is analogous to [MjLab](https://github.com/mujocolab/mjlab), but that project only supports the MuJoCo Warp backend — this one targets CPU MuJoCo and MJX (JAX-native, GPU-ready). Keep all design decisions consistent with this goal: declarative env config via dataclasses, Isaac Lab-style manager architecture, and pure-JAX JIT for observations/rewards/terminations.

## Workflow

- Git commit after large changes or new features.
- Use ruff for formatting and linting (`ruff format` + `ruff check`).
- After large changes, update README.md to reflect the latest state — but first summarize the planned changes to the user and wait for approval before editing.

## Commands

```bash
# Install (both steps required for test imports)
uv sync && uv pip install -e .

# Test all
uv run pytest

# Test single file or test
uv run pytest tests/test_env.py
uv run pytest tests/test_env.py::test_step

# Skip MJX tests if mujoco-mjx not installed
uv run pytest --ignore=tests/test_mjx.py

# Lint / format / typecheck
uv run ruff check src/ tests/ examples/
uv run ruff format src/ tests/ examples/
uv run pyright src/

# Run view/train scripts
uv run python scripts/view.py cartpole --rgb --steps 200
uv run python scripts/train.py cartpole --timesteps 500_000 --envs 4
mjpython scripts/view.py cartpole          # interactive viewer on macOS (requires mjpython)
```

## Architecture

### Two backends, one API

`ManagerBasedRlEnv` (CPU backend) and `MjxManagerBasedRlEnv` (MJX backend) share the same `ManagerBasedRlEnvCfg`. Both are `gymnasium.Env`.

**CPU backend step flow:**
1. `action_manager.process_actions()` → write to mjData ctrl
2. `sim.step()` × decimation (C, no JAX)
3. `extract_state()` — stack numpy arrays from all mjData into a `SimState` JAX pytree (the only numpy→JAX boundary)
4. `obs_manager.compute(state)`, `reward_manager.compute(state)`, `termination_manager.compute(state)` — all JIT-compiled JAX
5. Reset terminated envs, fire event functions (back to C)

**MJX backend:** the entire step (physics via `lax.scan` × decimation, obs, reward, done, conditional reset) is a single `@jax.jit` kernel — one JAX→numpy boundary per `step()` call.

### SimState — the JAX pytree boundary

`SimState` (`src/mjlabcpu/sim/sim_state.py`) is a frozen dataclass registered as a JAX pytree. All JIT-compiled managers receive and return `SimState`. Fields:
- `qpos (N, nq)`, `qvel (N, nv)`, `xpos (N, nbody, 3)`, `xquat (N, nbody, 4)`, `cvel (N, nbody, 6)`, `sensordata (N, nsensordata)`
- `episode_length (N,)`, `action (N, act_dim)`, `prev_action (N, act_dim)`, `commands dict[str, ndarray]`

**MuJoCo cvel convention:** `cvel[..., 0:3]` = angular velocity, `cvel[..., 3:6]` = linear velocity. Quaternions are wxyz throughout.

### SceneEntityCfg → ResolvedEntityCfg

`SceneEntityCfg(name="robot", joint_names=[...])` is declared in configs. At manager init time (before JIT), `.resolve(scene)` converts names to concrete `jnp.ndarray` index arrays (`qpos_addrs`, `qvel_addrs`, `body_ids`, etc.) stored in `ResolvedEntityCfg`. These arrays are captured as static data in JIT closures — never traced.

### Manager configuration pattern

All five managers are configured via dicts of term configs in `ManagerBasedRlEnvCfg`:

```python
actions={"cart": ActionTermCfg(cls=JointPositionAction, params={"entity_cfg": entity_cfg, "scale": 1.0})}
rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)}
terminations={"timeout": TerminationTermCfg(func=term_mdp.time_out, params={"max_episode_length": 1250}, time_out=True)}
observations={"policy": ObservationGroupCfg(terms={"joint_pos": ObservationTermCfg(func=obs_mdp.joint_pos_rel, params={"entity_cfg": entity_cfg})})}
events={"reset": EventTermCfg(func=event_mdp.reset_joints_uniform, mode="reset", params={...})}
```

`ActionTermCfg` takes `cls` + `params` dict — do **not** create a subclass. Action terms read config from `cfg.params` in `__init__`.

### Termination semantics

- `TerminationTermCfg(time_out=True)` → sets `truncated=True` (timeout)
- `TerminationTermCfg(time_out=False)` → sets `terminated=True` (failure)
- `terminated = done & ~truncated` in the env step

### Writing custom MDP functions

All obs/reward/termination functions are pure JAX with the same shape contract:

```python
def my_obs(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:  # (N, d)
def my_reward(state: SimState) -> jnp.ndarray:                               # (N,)
def my_termination(state: SimState, threshold: float) -> jnp.ndarray:        # (N,) bool
```

Extra parameters beyond `state` are bound via `functools.partial` at manager init. No Python control flow on traced values.

### Adding a new environment

Create a module in `examples/envs/` exposing:
- `make_env(num_envs: int, render_mode: str | None) -> ManagerBasedRlEnv`
- `ppo_cfg(num_envs: int) -> PPOCfg` (optional — used by `scripts/train.py`)

Then pass the env name to `scripts/view.py` or `scripts/train.py`.

## Key Constraints and Gotchas

- **Package install:** `uv sync --extra dev` does NOT install the package for test imports. Always run `uv pip install -e .` after `uv sync`.
- **MjSpec.attach:** Must pass a frame: `frame = spec.worldbody.add_frame(); spec.attach(child, prefix="name/", frame=frame)`.
- **macOS interactive viewer:** `launch_passive` requires `mjpython` — cannot run via VS Code debugpy or regular `python`. Must be launched from the terminal: `mjpython scripts/view.py <env>`. Offscreen `rgb_array` works with regular `python`.
- **JAX indexing:** `.at[env_ids].set(...)` requires `jnp.array(env_ids)` when `env_ids` is a plain Python list.
- **`render()` import:** Use `from mujoco import viewer as _mjviewer` inside functions to avoid shadowing the top-level `mujoco` import.
- **PPO update step:** Use plain `@jax.jit` — do NOT mark array args as `static_argnums`.
- **`flatten_rollout`** returns 6 values: `obs, actions, log_probs, advantages, returns, values`.
- **`joint_pos_limit()`** in terminations raises `NotImplementedError`.
- **`env.is_viewer_running()`** is the public API (not `env._viewer.is_running()`).
- **`UniformVelocityCommandCfg`** field is `resampling_time` (not `resampling_time_s`).
