"""EntityData — JAX state accessors for an Entity computed from SimState."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from mjlabcpu.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from mjlabcpu.entity.entity import Entity
    from mjlabcpu.sim.sim_state import SimState


class EntityData:
    """Convenience view of a :class:`SimState` sliced for a specific entity.

    All properties are pure JAX expressions so they are JIT-compatible.
    Instantiate once; the indexing arrays are static.
    """

    def __init__(self, entity: Entity) -> None:
        self._idx = entity.indexing

    # ------------------------------------------------------------------
    # Body state
    # ------------------------------------------------------------------

    def root_pos_w(self, state: SimState) -> jnp.ndarray:
        """Root body position in world frame. Shape: (num_envs, 3)."""
        return state.xpos[:, self._idx.root_body_id, :]

    def root_quat_w(self, state: SimState) -> jnp.ndarray:
        """Root body quaternion (wxyz) in world frame. Shape: (num_envs, 4)."""
        return state.xquat[:, self._idx.root_body_id, :]

    def root_lin_vel_w(self, state: SimState) -> jnp.ndarray:
        """Root body linear velocity in world frame. Shape: (num_envs, 3).

        MuJoCo cvel: [ang(0:3) | lin(3:6)] in body frame expressed in world coords.
        """
        return state.cvel[:, self._idx.root_body_id, 3:6]

    def root_ang_vel_w(self, state: SimState) -> jnp.ndarray:
        """Root body angular velocity in world frame. Shape: (num_envs, 3)."""
        return state.cvel[:, self._idx.root_body_id, 0:3]

    def root_lin_vel_b(self, state: SimState) -> jnp.ndarray:
        """Root body linear velocity in body (local) frame. Shape: (num_envs, 3)."""
        vel_w = self.root_lin_vel_w(state)
        quat_w = self.root_quat_w(state)
        return quat_rotate_inverse(quat_w, vel_w)

    def root_ang_vel_b(self, state: SimState) -> jnp.ndarray:
        """Root body angular velocity in body (local) frame. Shape: (num_envs, 3)."""
        ang_w = self.root_ang_vel_w(state)
        quat_w = self.root_quat_w(state)
        return quat_rotate_inverse(quat_w, ang_w)

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def joint_pos(self, state: SimState) -> jnp.ndarray:
        """Joint positions for this entity. Shape: (num_envs, nq)."""
        return state.qpos[:, self._idx.qpos_addrs]

    def joint_vel(self, state: SimState) -> jnp.ndarray:
        """Joint velocities for this entity. Shape: (num_envs, nv)."""
        return state.qvel[:, self._idx.qvel_addrs]

    def joint_pos_rel(self, state: SimState) -> jnp.ndarray:
        """Joint positions relative to default. Shape: (num_envs, nq)."""
        return self.joint_pos(state) - self._idx.default_qpos

    def joint_vel_rel(self, state: SimState) -> jnp.ndarray:
        """Joint velocities relative to default (always 0). Shape: (num_envs, nv)."""
        return self.joint_vel(state)

    # ------------------------------------------------------------------
    # Body state arrays
    # ------------------------------------------------------------------

    def body_pos_w(self, state: SimState) -> jnp.ndarray:
        """All body positions in world frame. Shape: (num_envs, nbody_entity, 3)."""
        return state.xpos[:, self._idx.body_ids, :]

    def body_quat_w(self, state: SimState) -> jnp.ndarray:
        """All body quaternions in world frame. Shape: (num_envs, nbody_entity, 4)."""
        return state.xquat[:, self._idx.body_ids, :]
