"""Pure JAX math utilities. All quaternions use MuJoCo's [w, x, y, z] (wxyz) convention."""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention)
# ---------------------------------------------------------------------------


@jax.jit
def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Return the conjugate of a quaternion [..., 4] in wxyz format."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


@jax.jit
def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions q1 * q2, both in wxyz format. Shape: [..., 4]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


@jax.jit
def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate vector v by quaternion q. q: [..., 4] wxyz, v: [..., 3]."""
    # Expand v as pure quaternion [0, vx, vy, vz]
    zero = jnp.zeros((*v.shape[:-1], 1))
    v_quat = jnp.concatenate([zero, v], axis=-1)
    rotated = quat_multiply(q, quat_multiply(v_quat, quat_conjugate(q)))
    return rotated[..., 1:]


@jax.jit
def quat_rotate_inverse(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate vector v by the inverse (conjugate) of quaternion q.

    Equivalent to expressing v in the frame defined by q.
    q: [..., 4] wxyz, v: [..., 3].
    """
    return quat_rotate(quat_conjugate(q), v)


@jax.jit
def quat_inv(q: jnp.ndarray) -> jnp.ndarray:
    """Unit-quaternion inverse = conjugate. q: [..., 4] wxyz."""
    return quat_conjugate(q)


@jax.jit
def euler_xyz_to_quat(euler: jnp.ndarray) -> jnp.ndarray:
    """Convert Euler XYZ angles (radians) to quaternion wxyz. Shape: [..., 3] → [..., 4]."""
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jnp.stack([w, x, y, z], axis=-1)


@jax.jit
def quat_to_euler_xyz(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion wxyz to Euler XYZ angles (radians). Shape: [..., 4] → [..., 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.stack([roll, pitch, yaw], axis=-1)


@jax.jit
def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jax.jit
def normalize(v: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """L2-normalize a vector along the last axis."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)


@jax.jit
def axis_angle_to_quat(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle to quaternion wxyz.

    axis: [..., 3] (unit vector), angle: [...] (radians).
    """
    half_angle = angle / 2.0
    w = jnp.cos(half_angle)
    xyz = axis * jnp.sin(half_angle)[..., None]
    return jnp.concatenate([w[..., None], xyz], axis=-1)


@jax.jit
def matrix_from_quat(q: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion wxyz to 3x3 rotation matrix. Shape: [..., 4] → [..., 3, 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    row0 = jnp.stack([r00, r01, r02], axis=-1)
    row1 = jnp.stack([r10, r11, r12], axis=-1)
    row2 = jnp.stack([r20, r21, r22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)
