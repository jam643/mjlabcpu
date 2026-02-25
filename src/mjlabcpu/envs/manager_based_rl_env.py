"""ManagerBasedRlEnv — top-level gymnasium-compatible RL environment."""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import jax.numpy as jnp
import mujoco
import numpy as np

from mjlabcpu.managers.action_manager import ActionManager, ActionTermCfg
from mjlabcpu.managers.command_manager import CommandManager, UniformVelocityCommandCfg
from mjlabcpu.managers.event_manager import EventManager, EventTermCfg
from mjlabcpu.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlabcpu.managers.reward_manager import RewardManager, RewardTermCfg
from mjlabcpu.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlabcpu.scene.scene import Scene, SceneCfg
from mjlabcpu.sim.sim import Simulation, SimulationCfg
from mjlabcpu.sim.sim_state import SimState, extract_state


@dataclasses.dataclass
class ManagerBasedRlEnvCfg:
    """Top-level environment configuration."""

    # --- Scene ---
    scene: SceneCfg = dataclasses.field(default_factory=SceneCfg)
    """Scene configuration (entities, ground plane, etc.)."""

    # --- Simulation ---
    sim: SimulationCfg = dataclasses.field(default_factory=SimulationCfg)
    """Physics simulation configuration."""

    # --- Episode ---
    episode_length_s: float = 20.0
    """Maximum episode duration in seconds."""
    decimation: int = 4
    """Number of physics steps per RL step."""

    # --- Managers ---
    observations: dict[str, ObservationGroupCfg] = dataclasses.field(default_factory=dict)
    rewards: dict[str, RewardTermCfg] = dataclasses.field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = dataclasses.field(default_factory=dict)
    actions: dict[str, ActionTermCfg] = dataclasses.field(default_factory=dict)
    events: dict[str, EventTermCfg] = dataclasses.field(default_factory=dict)
    commands: dict[str, UniformVelocityCommandCfg] = dataclasses.field(default_factory=dict)

    @property
    def max_episode_length(self) -> int:
        """Maximum number of RL steps per episode."""
        return int(self.episode_length_s / (self.sim.dt * self.decimation))


class ManagerBasedRlEnv(gym.Env):
    """Isaac Lab-style manager-based RL environment on CPU MuJoCo + JAX JIT.

    Step flow::

        step(actions):
          1. action_manager.process_actions(actions)
          2. for _ in decimation:
               action_manager.apply_actions()   # write ctrl to mjData (C)
               sim.step()                        # parallel mj_step (C, no JIT)
          3. state = extract_state(sim, ...)    # numpy → JAX array
          4. obs = obs_manager.compute(state)   # JIT-compiled JAX
          5. rewards = reward_manager.compute(state)    # JIT-compiled JAX
          6. done = termination_manager.compute(state)  # JIT-compiled JAX
          7. _reset_envs(done_ids)              # writes to mjData (C)
          8. return np.array(obs), ...

    Rendering notes:
        - ``render_mode="human"``: opens a passive MuJoCo viewer on env 0.
          On macOS this requires launching via ``mjpython script.py``.
        - ``render_mode="rgb_array"``: uses ``mujoco.Renderer`` for offscreen
          rendering; works with regular ``python``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: ManagerBasedRlEnvCfg, render_mode: str | None = None) -> None:
        self.cfg = cfg
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        # --- Build scene and compile model ---
        self._scene = Scene(cfg.scene)
        model = self._scene.compile()

        # --- Build simulation ---
        self._sim = Simulation(model, cfg.scene.num_envs, cfg.sim)
        self._sim.reset_all()

        # --- Episode tracking ---
        self._episode_length = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self._action = jnp.zeros((self.num_envs, 0))  # updated after action_manager init
        self._prev_action = jnp.zeros((self.num_envs, 0))

        # --- Build managers ---
        self._action_manager = ActionManager(cfg.actions, self)
        # Re-init action arrays with correct dim
        action_dim = self._action_manager.action_dim
        self._action = jnp.zeros((self.num_envs, action_dim))
        self._prev_action = jnp.zeros((self.num_envs, action_dim))

        self._command_manager = CommandManager(cfg.commands, self)
        self._obs_manager = ObservationManager(cfg.observations, self)
        self._reward_manager = RewardManager(cfg.rewards, self)
        self._termination_manager = TerminationManager(cfg.terminations, self)
        self._event_manager = EventManager(cfg.events, self)

        # --- JIT warm-up (trigger compilation before training loop) ---
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
    def sim(self) -> Simulation:
        return self._sim

    @property
    def max_episode_length(self) -> int:
        return self.cfg.max_episode_length

    @property
    def dt(self) -> float:
        """RL step duration in seconds."""
        return self._sim.dt * self.cfg.decimation

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset all environments."""
        super().reset(seed=seed)
        all_ids = list(range(self.num_envs))

        # Reset physics
        self._sim.reset_all()
        self._episode_length = jnp.zeros(self.num_envs, dtype=jnp.int32)

        # Reset action manager internal state (e.g. JointPosDeltaAction targets)
        # and seed self._action from the post-reset targets so last_action in
        # the first observation matches the actual position command.
        self._action_manager.reset(all_ids)
        self._action = self._action_manager.get_observed_actions()
        self._prev_action = self._action

        # Fire reset events
        self._event_manager.apply_reset(all_ids)

        # Resample commands
        self._command_manager.reset(all_ids)

        # Compute initial state
        state = extract_state(
            self._sim,
            self._action,
            self._prev_action,
            self._episode_length,
            self._command_manager.commands,
        )

        obs_dict = self._obs_manager.compute(state)
        obs = self._flatten_obs(obs_dict)
        return obs, {}

    def step(
        self, actions: np.ndarray | jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one RL step.

        Args:
            actions: (num_envs, action_dim) array.

        Returns:
            obs: (num_envs, obs_dim) observation array.
            rewards: (num_envs,) reward array.
            terminated: (num_envs,) bool — episode ended due to failure.
            truncated: (num_envs,) bool — episode ended due to timeout.
            info: dict with per-term reward and termination breakdowns.
        """
        # 1. Process actions
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)
        self._prev_action = self._action
        self._action_manager.process_actions(actions_jax)
        # Use observed_actions so stateful terms (e.g. JointPosDeltaAction)
        # expose their absolute target rather than the raw delta.
        self._action = self._action_manager.get_observed_actions()

        # 2. Physics loop (decimation)
        for _ in range(self.cfg.decimation):
            self._action_manager.apply_actions()
            self._sim.step()

        # 3. Update command manager
        self._command_manager.step(self.dt)

        # 4. Extract state (numpy → JAX)
        self._episode_length = self._episode_length + 1
        state = extract_state(
            self._sim,
            self._action,
            self._prev_action,
            self._episode_length,
            self._command_manager.commands,
        )

        # 5. JIT-compiled compute
        obs_dict = self._obs_manager.compute(state)
        total_reward, reward_terms = self._reward_manager.compute(state)
        done, truncated, term_terms = self._termination_manager.compute(state)

        # 6. Reset terminated environments
        terminated = done & ~truncated  # failure (not timeout)
        done_ids = np.where(np.array(done))[0].tolist()
        if done_ids:
            self._reset_envs(done_ids)

        # 7. Flatten observations
        obs = self._flatten_obs(obs_dict)

        info = {
            "reward_terms": {k: np.array(v) for k, v in reward_terms.items()},
            "termination_terms": {k: np.array(v) for k, v in term_terms.items()},
        }

        return (
            obs,
            np.array(total_reward),
            np.array(terminated),
            np.array(truncated),
            info,
        )

    def is_viewer_running(self) -> bool:
        """Return True if the passive viewer window is open."""
        return self._viewer is not None and self._viewer.is_running()

    def render(self) -> np.ndarray | None:
        """Render env 0.

        Returns:
            ``None`` for ``"human"`` mode (viewer runs in background thread).
            ``np.ndarray`` of shape ``(480, 640, 3)`` uint8 for ``"rgb_array"`` mode.
        """
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as _mjviewer

                self._viewer = _mjviewer.launch_passive(self._sim.model, self._sim.data[0])
            if self._viewer.is_running():
                with self._viewer.lock():
                    self._viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._sim.model, height=480, width=640)
            self._renderer.update_scene(self._sim.data[0])
            return self._renderer.render()
        return None

    def close(self) -> None:
        """Clean up resources."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._sim.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids: list[int]) -> None:
        """Reset specific environments and fire events."""
        # Reset episode tracking — use jnp.where to avoid numpy round-trip
        reset_mask = jnp.zeros(self.num_envs, dtype=jnp.bool_).at[jnp.array(env_ids)].set(True)
        self._episode_length = jnp.where(reset_mask, 0, self._episode_length)

        action = np.array(self._action)
        prev_action = np.array(self._prev_action)
        action[env_ids] = 0.0
        prev_action[env_ids] = 0.0
        self._action = jnp.array(action)
        self._prev_action = jnp.array(prev_action)

        # Reset action term internal state (e.g. JointPosDeltaAction targets)
        self._action_manager.reset(env_ids)

        # Fire reset events (writes to mjData, not JIT)
        self._event_manager.apply_reset(env_ids)

        # Resample commands
        self._command_manager.resample(env_ids)

        # Forward kinematics after reset
        for i in env_ids:
            mujoco.mj_forward(self._sim.model, self._sim.data[i])

    def _flatten_obs(self, obs_dict: dict[str, jnp.ndarray]) -> np.ndarray:
        """Concatenate all observation groups into a single array."""
        if not obs_dict:
            return np.zeros((self.num_envs, 0), dtype=np.float32)
        parts = list(obs_dict.values())
        if len(parts) == 1:
            return np.array(parts[0], dtype=np.float32)
        return np.array(jnp.concatenate(parts, axis=-1), dtype=np.float32)

    def _build_obs_space(self) -> gym.spaces.Box:
        """Compute observation space by running a dummy forward pass."""
        # Run a quick dummy state through obs manager to get shape
        dummy_state = self._make_dummy_state()
        obs_dict = self._obs_manager.compute(dummy_state)
        obs = self._flatten_obs(obs_dict)
        obs_dim = obs.shape[-1]
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _make_dummy_state(self) -> SimState:
        """Create a dummy SimState for shape inference."""
        return extract_state(
            self._sim,
            self._action,
            self._prev_action,
            self._episode_length,
            self._command_manager.commands,
        )

    def _warmup(self) -> None:
        """Trigger JIT compilation before training starts."""
        dummy = self._make_dummy_state()
        # Warm up all JIT-compiled managers
        self._obs_manager.compute(dummy)
        self._reward_manager.compute(dummy)
        self._termination_manager.compute(dummy)
