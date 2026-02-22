"""SimState — a JAX pytree snapshot of the simulation state."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from mjlabcpu.sim.sim import Simulation


@dataclasses.dataclass
class SimState:
    """Snapshot of MuJoCo simulation state as JAX arrays.

    Extracted from all mjData instances after each physics step.
    All arrays have leading batch dimension: (num_envs, ...).

    MuJoCo cvel ordering: [:, :, 0:3] = angular velocity, [:, :, 3:6] = linear velocity.
    """

    qpos: jnp.ndarray  # (num_envs, nq)
    qvel: jnp.ndarray  # (num_envs, nv)
    xpos: jnp.ndarray  # (num_envs, nbody, 3)  — body Cartesian positions
    xquat: jnp.ndarray  # (num_envs, nbody, 4)  — body quaternions (wxyz)
    cvel: jnp.ndarray  # (num_envs, nbody, 6)  — body spatial velocities [ang|lin]
    sensordata: jnp.ndarray  # (num_envs, nsensordata)
    episode_length: jnp.ndarray  # (num_envs,) int32 — steps since last reset
    action: jnp.ndarray  # (num_envs, action_dim)
    prev_action: jnp.ndarray  # (num_envs, action_dim)
    commands: dict[str, jnp.ndarray]  # named command tensors


# Register SimState as a JAX pytree so jax.jit can trace through it.
jax.tree_util.register_dataclass(
    SimState,
    data_fields=[
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "cvel",
        "sensordata",
        "episode_length",
        "action",
        "prev_action",
        "commands",
    ],
    meta_fields=[],
)


def extract_state_mjx(
    mjx_data: Any,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    episode_length: jnp.ndarray,
    commands: dict[str, jnp.ndarray],
) -> SimState:
    """Build a :class:`SimState` directly from batched ``mjx.Data``. Pure JAX.

    Unlike :func:`extract_state`, no numpy-to-JAX conversion is needed — all
    ``mjx.Data`` fields are already JAX arrays. JIT-compatible.
    """
    return SimState(
        qpos=mjx_data.qpos,
        qvel=mjx_data.qvel,
        xpos=mjx_data.xpos,
        xquat=mjx_data.xquat,
        cvel=mjx_data.cvel,
        sensordata=mjx_data.sensordata,
        episode_length=episode_length,
        action=action,
        prev_action=prev_action,
        commands=commands,
    )


def extract_state(
    sim: Simulation,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    episode_length: jnp.ndarray,
    commands: dict[str, jnp.ndarray],
) -> SimState:
    """Stack mjData arrays from all envs into a :class:`SimState` pytree.

    Called each RL step *after* ``sim.step()``.
    The numpy → JAX conversion happens here; physics is never touched by JAX.
    """
    data = sim.data
    return SimState(
        qpos=jnp.asarray(np.stack([d.qpos for d in data])),
        qvel=jnp.asarray(np.stack([d.qvel for d in data])),
        xpos=jnp.asarray(np.stack([d.xpos for d in data])),
        xquat=jnp.asarray(np.stack([d.xquat for d in data])),
        cvel=jnp.asarray(np.stack([d.cvel for d in data])),
        sensordata=jnp.asarray(np.stack([d.sensordata for d in data])),
        episode_length=episode_length,
        action=action,
        prev_action=prev_action,
        commands=commands,
    )
