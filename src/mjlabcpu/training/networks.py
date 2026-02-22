"""Actor-critic neural network for PPO using Flax."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    hidden_sizes: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.tanh(x)
        return nn.Dense(self.output_dim)(x)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs sharing the same module.

    Outputs:
        actor_mean: (batch, action_dim) — mean of Gaussian policy
        log_std:    (action_dim,)       — learnable log std (not batched)
        value:      (batch,)            — critic value estimate
    """

    hidden_sizes: tuple[int, ...]
    action_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        actor_mean = MLP(self.hidden_sizes, self.action_dim)(obs)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        value = MLP(self.hidden_sizes, 1)(obs).squeeze(-1)
        return actor_mean, log_std, value
