"""Command manager — generates and tracks velocity / goal commands."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from mjlabcpu.managers.manager_base import ManagerBase

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class UniformVelocityCommandCfg:
    """Uniform random velocity command configuration."""

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    """[min, max] linear x velocity."""
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    """[min, max] linear y velocity."""
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    """[min, max] angular z velocity."""
    resampling_time: float = 10.0
    """Seconds between command resampling."""


@dataclasses.dataclass
class GoalPositionCommandCfg:
    """Uniform random XY goal position command configuration.

    The goal is resampled on episode reset only (no time-based resampling).
    The command tensor has shape (num_envs, 3): [x, y, z].
    """

    x_range: tuple[float, float] = (-0.4, 0.4)
    """[min, max] goal x position."""
    y_range: tuple[float, float] = (-0.4, 0.4)
    """[min, max] goal y position."""
    z: float = 0.02
    """Fixed goal z height (e.g. puck centre above ground)."""


class CommandManager(ManagerBase):
    """Generates and maintains command tensors for all environments.

    Commands are stored as JAX arrays in a dict and passed into
    :class:`~mjlabcpu.sim.sim_state.SimState` each step.
    """

    def __init__(
        self,
        cfg: dict[str, UniformVelocityCommandCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._commands: dict[str, jnp.ndarray] = {}
        self._time_since_resample: dict[str, np.ndarray] = {}
        self._init_commands()

    def _init_commands(self) -> None:
        num_envs = self._env.num_envs
        for name, cmd_cfg in self._cfg.items():
            if isinstance(cmd_cfg, UniformVelocityCommandCfg):
                self._commands[name] = jnp.zeros((num_envs, 3))
            elif isinstance(cmd_cfg, GoalPositionCommandCfg):
                self._commands[name] = jnp.zeros((num_envs, 3))
            else:
                self._commands[name] = jnp.zeros((num_envs, 1))
            self._time_since_resample[name] = np.zeros(num_envs)

    def resample(self, env_ids: list[int]) -> None:
        """Resample commands for the given environments."""
        for name, cmd_cfg in self._cfg.items():
            if isinstance(cmd_cfg, UniformVelocityCommandCfg):
                cmd = np.array(self._commands[name])
                n = len(env_ids)
                cmd[env_ids, 0] = np.random.uniform(
                    cmd_cfg.lin_vel_x[0], cmd_cfg.lin_vel_x[1], n
                )
                cmd[env_ids, 1] = np.random.uniform(
                    cmd_cfg.lin_vel_y[0], cmd_cfg.lin_vel_y[1], n
                )
                cmd[env_ids, 2] = np.random.uniform(
                    cmd_cfg.ang_vel_z[0], cmd_cfg.ang_vel_z[1], n
                )
                self._commands[name] = jnp.array(cmd)
                self._time_since_resample[name][env_ids] = 0.0
            elif isinstance(cmd_cfg, GoalPositionCommandCfg):
                cmd = np.array(self._commands[name])
                n = len(env_ids)
                cmd[env_ids, 0] = np.random.uniform(
                    cmd_cfg.x_range[0], cmd_cfg.x_range[1], n
                )
                cmd[env_ids, 1] = np.random.uniform(
                    cmd_cfg.y_range[0], cmd_cfg.y_range[1], n
                )
                cmd[env_ids, 2] = cmd_cfg.z
                self._commands[name] = jnp.array(cmd)
                self._time_since_resample[name][env_ids] = 0.0

    def step(self, dt: float) -> None:
        """Update time counters and resample if interval exceeded."""
        num_envs = self._env.num_envs
        for name, cmd_cfg in self._cfg.items():
            if isinstance(cmd_cfg, UniformVelocityCommandCfg):
                self._time_since_resample[name] += dt
                resample_ids = np.where(
                    self._time_since_resample[name] >= cmd_cfg.resampling_time
                )[0].tolist()
                if resample_ids:
                    self.resample(resample_ids)

    @property
    def commands(self) -> dict[str, jnp.ndarray]:
        return self._commands

    def reset(self, env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        self.resample(env_ids)
