"""Pure JAX termination functions. All functions are JIT-compatible.

Signature: ``func(state: SimState, **static_params) → jnp.ndarray[bool]``
Shape: ``(num_envs,)`` — True means this term triggers termination.
"""

from __future__ import annotations

import jax.numpy as jnp

from mjlabcpu.managers.scene_entity_cfg import ResolvedEntityCfg
from mjlabcpu.sim.sim_state import SimState
from mjlabcpu.utils.math import quat_rotate_inverse


def time_out(state: SimState, max_episode_length: int) -> jnp.ndarray:
    """Terminate when episode length exceeds ``max_episode_length``. Shape: (num_envs,).

    This is a *timeout* (truncation), not a failure.
    """
    return state.episode_length >= max_episode_length


def bad_orientation(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    limit_angle: float = 1.0,
) -> jnp.ndarray:
    """Terminate when root body tilts beyond ``limit_angle`` radians. Shape: (num_envs,).

    Computed as the angle between the body z-axis and the world z-axis.
    """
    root_id = entity_cfg.root_body_id
    quat_w = state.xquat[:, root_id, :]
    # Project world up [0,0,1] into body frame
    up_w = jnp.zeros((*quat_w.shape[:-1], 3)).at[..., 2].set(1.0)
    up_b = quat_rotate_inverse(quat_w, up_w)  # (num_envs, 3)
    # Dot product of body z-axis with world up — 1.0 = perfectly upright, -1.0 = inverted
    dot_up = jnp.clip(up_b[:, 2], -1.0, 1.0)
    return dot_up < jnp.cos(limit_angle)


def root_height_below_minimum(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    minimum_height: float = 0.1,
) -> jnp.ndarray:
    """Terminate when root body z-position falls below ``minimum_height``. Shape: (num_envs,)."""
    root_id = entity_cfg.root_body_id
    height = state.xpos[:, root_id, 2]  # (num_envs,)
    return height < minimum_height


def joint_pos_limit(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    soft_ratio: float = 1.0,
) -> jnp.ndarray:
    """Terminate when any joint exceeds position limits. Shape: (num_envs,).

    Note: requires joint limits to be defined in the MJCF.

    .. warning::
        Not yet implemented. Raises ``NotImplementedError`` to prevent silent
        misconfiguration (previously returned all-False unconditionally).
    """
    raise NotImplementedError(
        "joint_pos_limit() is not yet implemented. "
        "Joint limit ranges (model.jnt_range) are not currently threaded through "
        "ResolvedEntityCfg. Remove this termination term or implement it manually."
    )


def object_out_of_bounds(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    max_xy_dist: float = 1.5,
) -> jnp.ndarray:
    """True if object root XY distance from origin exceeds max_xy_dist. Shape: (num_envs,) bool."""
    pos_xy = state.xpos[:, entity_cfg.root_body_id, :2]
    return jnp.sqrt(jnp.sum(pos_xy**2, axis=-1)) > max_xy_dist


def cartpole_fallen(
    state: SimState,
    entity_cfg: ResolvedEntityCfg,
    max_pole_angle: float = 0.2,
    max_cart_pos: float = 2.4,
) -> jnp.ndarray:
    """Terminate cartpole when pole falls or cart goes out of bounds. Shape: (num_envs,)."""
    qpos = state.qpos[:, entity_cfg.qpos_addrs]
    cart_pos = qpos[:, 0]  # slider position
    pole_angle = qpos[:, 1]  # hinge angle
    out_of_bounds = jnp.abs(cart_pos) > max_cart_pos
    pole_fallen = jnp.abs(pole_angle) > max_pole_angle
    return out_of_bounds | pole_fallen
