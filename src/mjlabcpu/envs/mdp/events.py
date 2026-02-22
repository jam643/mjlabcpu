"""Event functions — reset and randomisation. NOT JIT-compiled (write to mjData C arrays).

Signature: ``func(env, env_ids, **params) → None``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


def reset_scene_to_default(
    env: ManagerBasedRlEnv,
    env_ids: list[int],
) -> None:
    """Reset selected environments to the model's default state (mj_resetData)."""
    for i in env_ids:
        mujoco.mj_resetData(env.sim.model, env.sim.data[i])
        mujoco.mj_forward(env.sim.model, env.sim.data[i])


def reset_root_state_uniform(
    env: ManagerBasedRlEnv,
    env_ids: list[int],
    entity_name: str,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset root body state with uniform random perturbations.

    Args:
        entity_name: Entity whose root state to randomize.
        pose_range: Dict with optional keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
            each mapping to a (min, max) tuple.
        velocity_range: Dict with optional keys 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'.
    """
    pose_range = pose_range or {}
    velocity_range = velocity_range or {}

    entity = env.scene[entity_name]
    model = env.sim.model
    qpos_addrs = np.array(entity.qpos_addrs)
    qvel_addrs = np.array(entity.qvel_addrs)

    for i in env_ids:
        data = env.sim.data[i]

        # First reset to default
        mujoco.mj_resetData(model, data)

        # Apply position perturbations (for free joint: first 7 qpos entries)
        if len(qpos_addrs) >= 7:  # free joint
            # Position
            for ax_idx, key in enumerate(["x", "y", "z"]):
                if key in pose_range:
                    lo, hi = pose_range[key]
                    data.qpos[qpos_addrs[ax_idx]] = np.random.uniform(lo, hi)
            # Orientation (euler to quat)
            euler = np.zeros(3)
            for ax_idx, key in enumerate(["roll", "pitch", "yaw"]):
                if key in pose_range:
                    lo, hi = pose_range[key]
                    euler[ax_idx] = np.random.uniform(lo, hi)
            # Convert euler to quat (wxyz)
            cr, sr = np.cos(euler[0] / 2), np.sin(euler[0] / 2)
            cp, sp = np.cos(euler[1] / 2), np.sin(euler[1] / 2)
            cy, sy = np.cos(euler[2] / 2), np.sin(euler[2] / 2)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            data.qpos[qpos_addrs[3] : qpos_addrs[3] + 4] = [w, x, y, z]

            # Velocity (free joint: 6 dofs)
            if len(qvel_addrs) >= 6:
                for ax_idx, key in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"]):
                    if key in velocity_range:
                        lo, hi = velocity_range[key]
                        data.qvel[qvel_addrs[ax_idx]] = np.random.uniform(lo, hi)

        mujoco.mj_forward(model, data)


def reset_joints_uniform(
    env: ManagerBasedRlEnv,
    env_ids: list[int],
    entity_name: str,
    position_range: tuple[float, float] = (-0.1, 0.1),
    velocity_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Reset joint positions and velocities with uniform noise around defaults.

    Args:
        position_range: (min, max) uniform noise added to default joint positions.
        velocity_range: (min, max) uniform noise for initial joint velocities.
    """
    entity = env.scene[entity_name]
    model = env.sim.model
    qpos_addrs = np.array(entity.qpos_addrs)
    qvel_addrs = np.array(entity.qvel_addrs)
    default_qpos = np.array(entity.default_qpos)

    for i in env_ids:
        data = env.sim.data[i]
        lo_p, hi_p = position_range
        lo_v, hi_v = velocity_range
        noise_pos = np.random.uniform(lo_p, hi_p, len(qpos_addrs))
        noise_vel = np.random.uniform(lo_v, hi_v, len(qvel_addrs))
        data.qpos[qpos_addrs] = default_qpos + noise_pos
        data.qvel[qvel_addrs] = noise_vel
        mujoco.mj_forward(model, data)
