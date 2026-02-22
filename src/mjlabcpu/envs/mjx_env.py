"""MjxManagerBasedRlEnv — fully JIT-fused env step via MuJoCo MJX."""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlabcpu.managers.action_manager import ActionManager, ActionTermCfg
from mjlabcpu.managers.command_manager import CommandManager, UniformVelocityCommandCfg
from mjlabcpu.managers.event_manager import EventManager, EventTermCfg
from mjlabcpu.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlabcpu.managers.reward_manager import RewardManager, RewardTermCfg
from mjlabcpu.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlabcpu.scene.scene import Scene
from mjlabcpu.sim.mjx_sim import MjxSimulation
from mjlabcpu.sim.sim_state import SimState, extract_state_mjx


class MjxManagerBasedRlEnv(gym.Env):
    """Isaac Lab-style manager-based RL environment backed by MuJoCo MJX.

    The entire env step — ctrl computation, physics (×decimation), observation,
    reward, termination, and conditional reset — is fused into a single
    ``@jax.jit`` kernel via ``jax.lax.scan`` and ``jax.vmap``. There is only
    one JAX→numpy boundary per gymnasium ``step()`` call.

    Step flow (all inside one JIT call)::

        _jit_step(mjx_data, action, ...):
          1. compute_ctrl(action)                       # pure JAX
          2. lax.scan(vmap(mjx.step) × decimation)     # fused physics
          3. extract_state_mjx(mjx_data)               # zero-copy SimState
          4. obs / reward / done  (JIT-compiled managers)
          5. _conditional_reset(done, ...)              # jnp.where reset + noise

    Rendering notes:
        Only ``"rgb_array"`` is supported (no passive viewer for MJX). Env 0's
        state is copied back to a C ``MjData`` for rendering.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, cfg: ManagerBasedRlEnvCfg, render_mode: str | None = None
    ) -> None:
        self.cfg = cfg
        self.render_mode = render_mode
        self._renderer = None

        # --- Build scene and compile model ---
        self._scene = Scene(cfg.scene)
        model = self._scene.compile()

        # --- Build MJX simulation ---
        self._sim = MjxSimulation(model, cfg.scene.num_envs, cfg.sim)
        self._sim.reset_all()

        # --- Episode tracking ---
        self._episode_length = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self._action = jnp.zeros((self.num_envs, 0))
        self._prev_action = jnp.zeros((self.num_envs, 0))
        self._key = jax.random.PRNGKey(42)

        # --- Build managers (same API as CPU env) ---
        self._action_manager = ActionManager(cfg.actions, self)
        action_dim = self._action_manager.action_dim
        self._action = jnp.zeros((self.num_envs, action_dim))
        self._prev_action = jnp.zeros((self.num_envs, action_dim))

        self._command_manager = CommandManager(cfg.commands, self)
        self._obs_manager = ObservationManager(cfg.observations, self)
        self._reward_manager = RewardManager(cfg.rewards, self)
        self._termination_manager = TerminationManager(cfg.terminations, self)
        self._event_manager = EventManager(cfg.events, self)

        # --- Build fused JIT step ---
        self._build_jit_step()

        # --- JIT warm-up ---
        self._warmup()

        # --- Gymnasium spaces ---
        self.observation_space = self._build_obs_space()
        self.action_space = self._action_manager.action_space

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.cfg.scene.num_envs

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def sim(self) -> MjxSimulation:
        return self._sim

    @property
    def max_episode_length(self) -> int:
        return self.cfg.max_episode_length

    @property
    def dt(self) -> float:
        return self._sim.dt * self.cfg.decimation

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._sim.reset_all()
        self._episode_length = jnp.zeros(self.num_envs, dtype=jnp.int32)
        action_dim = self._action_manager.action_dim
        self._action = jnp.zeros((self.num_envs, action_dim))
        self._prev_action = jnp.zeros((self.num_envs, action_dim))
        self._key = jax.random.PRNGKey(seed if seed is not None else 42)
        self._command_manager.reset(list(range(self.num_envs)))

        state = self._extract_state_now()
        obs_dict = self._obs_manager.compute(state)
        obs = self._flatten_obs_jax(obs_dict)
        return np.array(obs, dtype=np.float32), {}

    def step(
        self, actions: np.ndarray | jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one RL step via the fused JIT kernel.

        The entire physics + obs + reward + done + reset computation is a
        single JIT-compiled JAX call. The only boundary crossing is the final
        ``np.array()`` conversion on return.
        """
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)

        # Update command manager (lightweight, outside JIT)
        self._command_manager.step(self.dt)

        (
            self._sim.mjx_data,
            obs_jax,
            total_reward,
            reward_terms,
            done,
            truncated,
            term_terms,
            self._episode_length,
            self._key,
        ) = self._jit_step(
            self._sim.mjx_data,
            actions_jax,
            self._prev_action,
            self._episode_length,
            self._command_manager.commands,
            self._key,
        )

        self._prev_action = self._action
        self._action = actions_jax
        terminated = done & ~truncated

        info = {
            "reward_terms": {k: np.array(v) for k, v in reward_terms.items()},
            "termination_terms": {k: np.array(v) for k, v in term_terms.items()},
        }
        return (
            np.array(obs_jax, dtype=np.float32),
            np.array(total_reward),
            np.array(terminated),
            np.array(truncated),
            info,
        )

    def render(self) -> np.ndarray | None:
        """Render env 0 via offscreen renderer (rgb_array only).

        Copies JAX state back to a temporary C MjData for rendering.
        """
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._sim.model, height=480, width=640)
        data = mujoco.MjData(self._sim.model)
        np.copyto(data.qpos, np.array(self._sim.mjx_data.qpos[0]))
        np.copyto(data.qvel, np.array(self._sim.mjx_data.qvel[0]))
        mujoco.mj_forward(self._sim.model, data)
        self._renderer.update_scene(data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._sim.close()

    # ------------------------------------------------------------------
    # JIT step construction
    # ------------------------------------------------------------------

    def _build_jit_step(self) -> None:
        """Build and JIT-compile the fused env step closure.

        Captures all static configuration (mjx model, manager functions,
        reset parameters, decimation count) into the closure so JAX can
        compile a single monolithic kernel.
        """
        mjx_model = self._sim.mjx_model
        init_mjx_data = self._sim.init_mjx_data  # single-env template
        decimation = self.cfg.decimation
        nu = self._sim.model.nu

        # Freeze references to manager compute functions
        obs_fn = self._obs_manager.compute
        reward_fn = self._reward_manager.compute
        term_fn = self._termination_manager.compute

        # Freeze action manager for ctrl computation
        action_manager = self._action_manager

        def compute_ctrl(action: jnp.ndarray) -> jnp.ndarray:
            """Pure JAX: flat action → (num_envs, nu) ctrl array."""
            ctrl = jnp.zeros((action.shape[0], nu))
            offset = 0
            for term in action_manager._terms.values():
                dim = term.action_dim
                ctrl = term.compute_ctrl_jax(ctrl, action[:, offset : offset + dim])
                offset += dim
            return ctrl

        # Parse reset parameters from event configs into concrete JAX arrays
        reset_ops = _parse_reset_ops(self.cfg.events, self._scene)

        def jit_step(
            mjx_data: Any,
            action: jnp.ndarray,
            prev_action: jnp.ndarray,
            episode_length: jnp.ndarray,
            commands: dict[str, jnp.ndarray],
            key: jnp.ndarray,
        ) -> tuple:
            # 1. Ctrl from action
            ctrl = compute_ctrl(action)  # (N, nu)

            # 2. Physics × decimation — vmap over envs, scan over steps
            def phys_step(data: Any, _: Any) -> tuple[Any, None]:
                def step_env(d: Any, c: jnp.ndarray) -> Any:
                    return mjx.step(mjx_model, d.replace(ctrl=c))

                return jax.vmap(step_env)(data, ctrl), None

            mjx_data, _ = jax.lax.scan(
                phys_step, mjx_data, None, length=decimation
            )

            # 3. State extraction — zero-copy from JAX pytree
            episode_length = episode_length + 1
            state = extract_state_mjx(
                mjx_data, action, prev_action, episode_length, commands
            )

            # 4. Obs / reward / done (manager JIT fns inline into this kernel)
            obs_dict = obs_fn(state)
            total_reward, reward_terms = reward_fn(state)
            done, truncated, term_terms = term_fn(state)

            # 5. Flatten obs inside JAX (stay in JAX until return)
            obs_parts = list(obs_dict.values())
            obs = obs_parts[0] if len(obs_parts) == 1 else jnp.concatenate(obs_parts, axis=-1)

            # 6. Conditional reset for done envs (jnp.where — no Python branching)
            mjx_data, episode_length, key = _conditional_reset(
                done, mjx_data, episode_length, key, init_mjx_data, reset_ops
            )

            return (
                mjx_data,
                obs,
                total_reward,
                reward_terms,
                done,
                truncated,
                term_terms,
                episode_length,
                key,
            )

        self._jit_step = jax.jit(jit_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_state_now(self) -> SimState:
        return extract_state_mjx(
            self._sim.mjx_data,
            self._action,
            self._prev_action,
            self._episode_length,
            self._command_manager.commands,
        )

    def _flatten_obs_jax(self, obs_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if not obs_dict:
            return jnp.zeros((self.num_envs, 0))
        parts = list(obs_dict.values())
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)

    def _build_obs_space(self) -> gym.spaces.Box:
        state = self._extract_state_now()
        obs_dict = self._obs_manager.compute(state)
        obs = self._flatten_obs_jax(obs_dict)
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32
        )

    def _warmup(self) -> None:
        """Trigger JIT compilation before the training loop."""
        state = self._extract_state_now()
        self._obs_manager.compute(state)
        self._reward_manager.compute(state)
        self._termination_manager.compute(state)
        dummy_action = jnp.zeros((self.num_envs, self._action_manager.action_dim))
        self._jit_step(
            self._sim.mjx_data,
            dummy_action,
            dummy_action,
            self._episode_length,
            self._command_manager.commands,
            self._key,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (called during _build_jit_step, results used in JIT)
# ---------------------------------------------------------------------------


def _parse_reset_ops(
    events_cfg: dict[str, EventTermCfg], scene: Scene
) -> list[tuple]:
    """Extract reset parameters from event configs into concrete JAX arrays.

    Returns a list of ``("joints_uniform", qpos_addrs, qvel_addrs,
    default_qpos, pos_range, vel_range)`` tuples baked from the event configs.
    Only :func:`~mjlabcpu.envs.mdp.events.reset_joints_uniform` is currently
    supported; other event functions are skipped.
    """
    from mjlabcpu.envs.mdp.events import reset_joints_uniform

    ops = []
    for term_cfg in events_cfg.values():
        if term_cfg.mode != "reset":
            continue
        if term_cfg.func is not reset_joints_uniform:
            continue

        params = term_cfg.params
        entity_name: str = params["entity_name"]
        pos_range: tuple[float, float] = params.get("position_range", (-0.1, 0.1))
        vel_range: tuple[float, float] = params.get("velocity_range", (-0.1, 0.1))

        entity = scene[entity_name]
        qpos_addrs = jnp.asarray(np.array(entity.qpos_addrs), dtype=jnp.int32)
        qvel_addrs = jnp.asarray(np.array(entity.qvel_addrs), dtype=jnp.int32)
        default_qpos = jnp.asarray(np.array(entity.default_qpos), dtype=jnp.float32)

        ops.append(
            ("joints_uniform", qpos_addrs, qvel_addrs, default_qpos, pos_range, vel_range)
        )

    return ops


def _conditional_reset(
    done: jnp.ndarray,
    mjx_data: Any,
    episode_length: jnp.ndarray,
    key: jnp.ndarray,
    init_mjx_data: Any,
    reset_ops: list[tuple],
) -> tuple[Any, jnp.ndarray, jnp.ndarray]:
    """Conditionally reset done environments inside JIT.

    For each done env:
      - All ``mjx.Data`` fields are reset to ``init_mjx_data`` via ``jnp.where``
      - Joint qpos/qvel are overwritten with ``default + uniform noise``

    For non-done envs: all fields are unchanged.

    Args:
        done:           (N,) bool — which envs to reset.
        mjx_data:       Batched ``mjx.Data`` (N-env pytree).
        episode_length: (N,) int32 step counters.
        key:            JAX PRNG key.
        init_mjx_data:  Single-env default ``mjx.Data`` (reset template).
        reset_ops:      List of reset parameter tuples from :func:`_parse_reset_ops`.

    Returns:
        ``(new_mjx_data, new_episode_length, new_key)``
    """

    def reset_leaf(d: jnp.ndarray, init: jnp.ndarray) -> jnp.ndarray:
        # d: (N, *s)   init: (*s,)   done: (N,)
        # Reshape done to broadcast over trailing dims
        done_bc = done.reshape((-1,) + (1,) * (d.ndim - 1))
        return jnp.where(done_bc, init, d)

    # Reset all fields for done envs
    new_mjx_data = jax.tree_util.tree_map(reset_leaf, mjx_data, init_mjx_data)

    # Apply randomised noise on top of the reset qpos/qvel
    new_qpos = new_mjx_data.qpos  # (N, nq) — already default for done envs
    new_qvel = new_mjx_data.qvel  # (N, nv) — already zero for done envs

    for _, qp_addrs, qv_addrs, default_qpos, pos_range, vel_range in reset_ops:
        key, k1, k2 = jax.random.split(key, 3)
        N = done.shape[0]
        nqp, nqv = qp_addrs.shape[0], qv_addrs.shape[0]

        noise_pos = jax.random.uniform(
            k1, (N, nqp), minval=pos_range[0], maxval=pos_range[1]
        )
        noise_vel = jax.random.uniform(
            k2, (N, nqv), minval=vel_range[0], maxval=vel_range[1]
        )

        # For done envs: default_qpos + noise; for non-done: keep current
        cur_qpos_at = new_qpos[:, qp_addrs]  # (N, nqp)
        cur_qvel_at = new_qvel[:, qv_addrs]  # (N, nqv)

        new_qpos_at = jnp.where(done[:, None], default_qpos[None] + noise_pos, cur_qpos_at)
        new_qvel_at = jnp.where(done[:, None], noise_vel, cur_qvel_at)

        new_qpos = new_qpos.at[:, qp_addrs].set(new_qpos_at)
        new_qvel = new_qvel.at[:, qv_addrs].set(new_qvel_at)

    new_mjx_data = new_mjx_data.replace(qpos=new_qpos, qvel=new_qvel)
    new_episode_length = jnp.where(done, jnp.zeros_like(episode_length), episode_length)

    return new_mjx_data, new_episode_length, key
