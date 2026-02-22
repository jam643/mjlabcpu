"""Pure JAX observation functions. All functions are JIT-compatible.

Signature: ``func(state: SimState, **static_params) → jnp.ndarray``
Shape: ``(num_envs, obs_dim)``

Static params are resolved at manager init time and baked into functools.partial closures.
"""

from __future__ import annotations

import jax.numpy as jnp

from mjlabcpu.managers.scene_entity_cfg import ResolvedEntityCfg
from mjlabcpu.sim.sim_state import SimState
from mjlabcpu.utils.math import quat_rotate_inverse


def base_lin_vel(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Root body linear velocity in body frame. Shape: (num_envs, 3).

    Uses :attr:`~mjlabcpu.sim.sim_state.SimState.cvel` which stores
    [ang(0:3) | lin(3:6)] for each body.
    """
    root_id = entity_cfg.root_body_id
    vel_w = state.cvel[:, root_id, 3:6]  # linear part
    quat_w = state.xquat[:, root_id, :]
    return quat_rotate_inverse(quat_w, vel_w)


def base_ang_vel(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Root body angular velocity in body frame. Shape: (num_envs, 3)."""
    root_id = entity_cfg.root_body_id
    ang_w = state.cvel[:, root_id, 0:3]  # angular part
    quat_w = state.xquat[:, root_id, :]
    return quat_rotate_inverse(quat_w, ang_w)


def projected_gravity(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Gravity vector projected into body frame. Shape: (num_envs, 3).

    Returns the unit gravity direction expressed in the robot's body frame,
    useful for orientation estimation without an IMU.
    """
    root_id = entity_cfg.root_body_id
    quat_w = state.xquat[:, root_id, :]
    # World gravity direction (unit): [0, 0, -1]
    g_w = jnp.zeros((*quat_w.shape[:-1], 3)).at[..., 2].set(-1.0)
    return quat_rotate_inverse(quat_w, g_w)


def joint_pos_rel(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Joint positions relative to defaults. Shape: (num_envs, nq).

    Uses :attr:`ResolvedEntityCfg.qpos_addrs` for indexing and
    :attr:`ResolvedEntityCfg.default_qpos` for the baseline.
    """
    return state.qpos[:, entity_cfg.qpos_addrs] - entity_cfg.default_qpos


def joint_vel_rel(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Joint velocities. Shape: (num_envs, nv)."""
    return state.qvel[:, entity_cfg.qvel_addrs]


def last_action(state: SimState) -> jnp.ndarray:
    """Last applied action. Shape: (num_envs, action_dim)."""
    return state.action


def generated_commands(state: SimState, command_name: str) -> jnp.ndarray:
    """Named command tensor from :attr:`SimState.commands`. Shape: (num_envs, cmd_dim)."""
    return state.commands[command_name]


def body_pos_w(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """All body positions in world frame for the entity. Shape: (num_envs, nbody*3)."""
    pos = state.xpos[:, entity_cfg.body_ids, :]  # (num_envs, nbody, 3)
    return pos.reshape(pos.shape[0], -1)


def root_pos_w(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Root body position in world frame. Shape: (num_envs, 3)."""
    return state.xpos[:, entity_cfg.root_body_id, :]


def root_quat_w(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Root body quaternion (wxyz) in world frame. Shape: (num_envs, 4)."""
    return state.xquat[:, entity_cfg.root_body_id, :]


def body_pos_w_xy(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """XY position of the first body in entity_cfg.body_ids. Shape: (num_envs, 2).

    Use with SceneEntityCfg(name="panda", body_names=["link7"]) for end-effector XY.
    """
    return state.xpos[:, entity_cfg.body_ids[0], :2]


def height_scan(
    state: SimState,
    sensor_slice: tuple[int, int] | None = None,
) -> jnp.ndarray:
    """Height scan from sensor data. Shape: (num_envs, n_sensors).

    Args:
        sensor_slice: (start, end) slice into sensordata. If None, use all.
    """
    if sensor_slice is not None:
        return state.sensordata[:, sensor_slice[0] : sensor_slice[1]]
    return state.sensordata
