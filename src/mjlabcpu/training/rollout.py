"""Rollout buffer for collecting PPO trajectories."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class RolloutBuffer:
    """Stores one rollout of trajectories across all environments.

    All arrays have leading shape ``(num_steps, num_envs)``.
    ``advantages`` and ``returns`` are filled in by :func:`compute_gae`.
    """

    obs: jnp.ndarray          # (T, num_envs, obs_dim)
    actions: jnp.ndarray      # (T, num_envs, act_dim)
    rewards: jnp.ndarray      # (T, num_envs)
    dones: jnp.ndarray        # (T, num_envs)  bool
    values: jnp.ndarray       # (T, num_envs)
    log_probs: jnp.ndarray    # (T, num_envs)
    advantages: jnp.ndarray   # (T, num_envs)
    returns: jnp.ndarray      # (T, num_envs)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalised Advantage Estimation (GAE).

    Args:
        rewards:    (T, num_envs)
        values:     (T, num_envs)
        dones:      (T, num_envs) bool — True at terminal step
        last_value: (num_envs,)   — V(s_{T+1})
        gamma:      discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T, num_envs)
        returns:    (T, num_envs)  — advantages + values (target for critic)
    """
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros(rewards.shape[1])

    for t in reversed(range(T)):
        if t == T - 1:
            # Mask last_value by whether each env terminated at the final step.
            # Auto-reset envs return the NEW episode's obs, so last_value contains
            # V(s_0_new_ep) which must be zeroed for true terminal steps.
            next_non_terminal = 1.0 - dones[T - 1].astype(jnp.float32)
            next_value = last_value * next_non_terminal
        else:
            next_non_terminal = 1.0 - dones[t].astype(jnp.float32)
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values
    return advantages, returns


def flatten_rollout(
    buf: RolloutBuffer,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flatten (T, num_envs, ...) → (T*num_envs, ...) for minibatch sampling.

    Returns:
        obs, actions, log_probs, advantages, returns, values
        Values are included for clipped value-function loss.
    """
    T, N = buf.obs.shape[:2]
    B = T * N
    obs = buf.obs.reshape(B, -1)
    actions = buf.actions.reshape(B, -1)
    log_probs = buf.log_probs.reshape(B)
    advantages = buf.advantages.reshape(B)
    returns = buf.returns.reshape(B)
    values = buf.values.reshape(B)
    return obs, actions, log_probs, advantages, returns, values
