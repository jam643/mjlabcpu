"""Pure JAX reward functions. All functions are JIT-compatible.

Signature: ``func(state: SimState, **static_params) → jnp.ndarray``
Shape: ``(num_envs,)`` — one scalar per environment.
"""

from __future__ import annotations

import jax.numpy as jnp

from mjlabcpu.managers.scene_entity_cfg import ResolvedEntityCfg
from mjlabcpu.sim.sim_state import SimState
from mjlabcpu.utils.math import quat_rotate_inverse


# ---------------------------------------------------------------------------
# Survival / existence rewards
# ---------------------------------------------------------------------------


def is_alive(state: SimState) -> jnp.ndarray:
    """+1 for every non-terminated step. Shape: (num_envs,)."""
    return jnp.ones(state.qpos.shape[0])


# ---------------------------------------------------------------------------
# Regularisation / penalty rewards
# ---------------------------------------------------------------------------


def joint_torques_l2(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Negative L2 norm of joint torques (using action as proxy). Shape: (num_envs,).

    Note: MuJoCo actuator forces require sensor data or qfrc_actuator. Here we
    use the processed action as a proxy for control effort.
    """
    action = state.action  # (num_envs, action_dim)
    return -jnp.sum(action**2, axis=-1)


def joint_vel_l2(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Negative L2 norm of joint velocities. Shape: (num_envs,)."""
    qvel = state.qvel[:, entity_cfg.qvel_addrs]
    return -jnp.sum(qvel**2, axis=-1)


def action_rate_l2(state: SimState) -> jnp.ndarray:
    """Negative L2 norm of action change (action smoothness). Shape: (num_envs,)."""
    diff = state.action - state.prev_action
    return -jnp.sum(diff**2, axis=-1)


def flat_orientation_l2(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Negative L2 norm of projected gravity deviation from [0,0,-1]. Shape: (num_envs,).

    Penalises tilting of the robot base.
    """
    root_id = entity_cfg.root_body_id
    quat_w = state.xquat[:, root_id, :]
    g_w = jnp.zeros((*quat_w.shape[:-1], 3)).at[..., 2].set(-1.0)
    g_b = quat_rotate_inverse(quat_w, g_w)
    # Penalise deviation from downward direction in body frame
    # [0,0,-1] in body frame = perfectly flat; penalise x,y components
    return -(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)


# ---------------------------------------------------------------------------
# Velocity tracking rewards
# ---------------------------------------------------------------------------


def track_lin_vel_xy(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    command_name: str,
    std: float = 0.25,
) -> jnp.ndarray:
    """Reward for tracking linear XY velocity command. Shape: (num_envs,).

    Uses Gaussian kernel: exp(-||v_cmd - v_actual||^2 / std^2).
    """
    root_id = entity_cfg.root_body_id
    vel_w = state.cvel[:, root_id, 3:6]  # (num_envs, 3) linear vel in world
    quat_w = state.xquat[:, root_id, :]
    vel_b = quat_rotate_inverse(quat_w, vel_w)  # (num_envs, 3) in body frame

    cmd = state.commands[command_name]  # (num_envs, 3) [vx, vy, wz]
    lin_vel_error = jnp.sum((cmd[:, :2] - vel_b[:, :2]) ** 2, axis=-1)
    return jnp.exp(-lin_vel_error / std**2)


def track_ang_vel_z(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    command_name: str,
    std: float = 0.25,
) -> jnp.ndarray:
    """Reward for tracking angular Z velocity command. Shape: (num_envs,)."""
    root_id = entity_cfg.root_body_id
    ang_w = state.cvel[:, root_id, 0:3]  # (num_envs, 3) angular vel in world
    quat_w = state.xquat[:, root_id, :]
    ang_b = quat_rotate_inverse(quat_w, ang_w)  # body frame

    cmd = state.commands[command_name]  # (num_envs, 3)
    ang_vel_error = (cmd[:, 2] - ang_b[:, 2]) ** 2
    return jnp.exp(-ang_vel_error / std**2)


def joint_pos_deviation(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Negative L2 norm of joint position deviation from defaults. Shape: (num_envs,)."""
    qpos = state.qpos[:, entity_cfg.qpos_addrs]
    return -jnp.sum((qpos - entity_cfg.default_qpos) ** 2, axis=-1)


def upright(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Reward for keeping the root body upright (z-axis aligned with world z). Shape: (num_envs,)."""
    root_id = entity_cfg.root_body_id
    quat_w = state.xquat[:, root_id, :]
    # Project world up-vector [0,0,1] into body frame
    up_w = jnp.zeros((*quat_w.shape[:-1], 3)).at[..., 2].set(1.0)
    up_b = quat_rotate_inverse(quat_w, up_w)
    # Cosine similarity with body z-axis [0,0,1]
    return up_b[:, 2]  # perfect upright = 1.0


def object_to_goal(
    state: SimState,
    object_entity_cfg: ResolvedEntityCfg,
    command_name: str,
) -> jnp.ndarray:
    """Reward = -||puck_xy - goal_xy||. Shape: (num_envs,).

    Encourages pushing the object toward the goal position.
    """
    obj_xy = state.xpos[:, object_entity_cfg.root_body_id, :2]
    goal_xy = state.commands[command_name][:, :2]
    return -jnp.sqrt(jnp.sum((obj_xy - goal_xy) ** 2, axis=-1) + 1e-6)


def eef_to_object(
    state: SimState,
    eef_entity_cfg: ResolvedEntityCfg,
    object_entity_cfg: ResolvedEntityCfg,
) -> jnp.ndarray:
    """Reward = -||eef_xy - puck_xy||. Shape: (num_envs,).

    Encourages the arm end-effector to approach the object.
    """
    eef_xy = state.xpos[:, eef_entity_cfg.body_ids[0], :2]
    obj_xy = state.xpos[:, object_entity_cfg.root_body_id, :2]
    return -jnp.sqrt(jnp.sum((eef_xy - obj_xy) ** 2, axis=-1) + 1e-6)


def cartpole_upright(state: SimState, entity_cfg: ResolvedEntityCfg) -> jnp.ndarray:
    """Reward for cartpole pole being upright. Uses cos(pole_angle). Shape: (num_envs,)."""
    # For cartpole: qpos = [slider, hinge_angle]
    qpos = state.qpos[:, entity_cfg.qpos_addrs]
    if qpos.shape[-1] < 2:
        raise ValueError(
            f"cartpole_upright() requires at least 2 joint DOFs (slider + hinge), "
            f"but entity '{entity_cfg.entity}' has qpos shape {qpos.shape}. "
            "Check your SceneEntityCfg configuration."
        )
    angle = qpos[:, 1]  # hinge angle (pole angle from upright)
    return jnp.cos(angle)
