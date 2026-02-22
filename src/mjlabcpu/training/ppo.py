"""Pure JAX PPO trainer for mjlabcpu environments."""

from __future__ import annotations

import dataclasses
import pickle
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from mjlabcpu.training.networks import ActorCritic
from mjlabcpu.training.rollout import RolloutBuffer, compute_gae, flatten_rollout


@dataclasses.dataclass
class PPOCfg:
    """Hyperparameters for the PPO trainer."""

    # Rollout
    num_steps: int = 2048
    """Number of environment steps collected per rollout (per env)."""
    num_envs: int = 4
    """Must match the env's ``num_envs``."""

    # PPO
    learning_rate: float = 3e-4
    num_epochs: int = 10
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    hidden_sizes: tuple[int, ...] = (256, 128)

    # Logging
    log_interval: int = 10
    """Log metrics every N PPO updates."""
    wandb_project: str | None = None
    """W&B project name. ``None`` disables W&B logging."""
    wandb_run_name: str | None = None
    """Optional W&B run name. Auto-generated if ``None``."""


# ---------------------------------------------------------------------------
# Helper: Gaussian log-prob and entropy
# ---------------------------------------------------------------------------

def _gaussian_log_prob(
    actions: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    """Log-prob of ``actions`` under N(mean, exp(log_std)^2). Returns (batch,)."""
    std = jnp.exp(log_std)
    log_prob = -0.5 * (
        ((actions - mean) / (std + 1e-8)) ** 2
        + 2 * log_std
        + jnp.log(2 * jnp.pi)
    )
    return log_prob.sum(axis=-1)


def _gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    """Differential entropy of N(0, diag(exp(log_std)^2)). Returns scalar.

    Uses mean over action dimensions so the coefficient is scale-invariant
    regardless of the number of actuators.
    """
    return (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).mean()


# ---------------------------------------------------------------------------
# PPO update step (JIT-compiled)
# ---------------------------------------------------------------------------

@jax.jit
def _ppo_update_step(
    state: TrainState,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """One gradient step on a minibatch.

    Args:
        state:         Flax TrainState (params + optimizer)
        obs:           (B, obs_dim)
        actions:       (B, act_dim)
        old_log_probs: (B,)
        old_values:    (B,) — critic values from rollout (for clipped value loss)
        advantages:    (B,) — normalised outside
        returns:       (B,) — critic targets

    Returns:
        updated TrainState and dict of scalar losses.
    """

    def loss_fn(params):
        actor_mean, log_std, values = state.apply_fn({"params": params}, obs)
        log_probs = _gaussian_log_prob(actions, actor_mean, log_std)
        entropy = _gaussian_entropy(log_std)

        # Policy loss (clipped surrogate)
        ratio = jnp.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss — clipped to prevent critic from diverging
        v_clipped = old_values + jnp.clip(values - old_values, -clip_coef, clip_coef)
        v_loss = 0.5 * jnp.maximum(
            (values - returns) ** 2, (v_clipped - returns) ** 2
        ).mean()

        total_loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
        return total_loss, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}

    (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, aux


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Minimal pure-JAX PPO trainer for mjlabcpu gymnasium environments.

    Example::

        trainer = PPOTrainer(env, PPOCfg(num_envs=4))
        metrics = trainer.train(total_timesteps=500_000)
        trainer.save("checkpoints/policy.pkl")
    """

    def __init__(self, env: Any, cfg: PPOCfg) -> None:
        self.env = env
        self.cfg = cfg

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Build network
        self._net = ActorCritic(
            hidden_sizes=cfg.hidden_sizes,
            action_dim=act_dim,
        )

        # Initialise params
        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = self._net.init(key, dummy_obs)["params"]

        # Optimiser with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.learning_rate),
        )
        self._state = TrainState.create(
            apply_fn=self._net.apply,
            params=params,
            tx=tx,
        )

        self._obs_dim = obs_dim
        self._act_dim = act_dim

        # JIT-compile the full inference step (forward pass + sampling + log_prob)
        # so the entire computation is one fused kernel call during rollout collection.
        net = self._net

        @jax.jit
        def _jit_act(
            params: Any, obs: jnp.ndarray, key: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """obs → (actions, log_probs, values, next_key) — JIT-compiled."""
            mean, log_std, values = net.apply({"params": params}, obs)
            std = jnp.exp(log_std)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, mean.shape)
            actions = mean + std * noise
            log_probs = _gaussian_log_prob(actions, mean, log_std)
            return actions, log_probs, values, key

        self._jit_act = _jit_act

        # Cached JIT for deterministic inference (avoids re-tracing on every call)
        self._jit_apply = jax.jit(self._net.apply)

        self._wandb = None
        if cfg.wandb_project is not None:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config=dataclasses.asdict(cfg),
                )
                self._wandb = wandb
                print(f"W&B logging → project={cfg.wandb_project!r}")
            except ImportError:
                print("W&B logging requested but wandb is not installed: pip install wandb")
            except Exception as e:
                print(f"W&B init failed ({e}). Training will continue without logging.")
                print("  Re-authenticate with: wandb login --relogin")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int) -> dict[str, list[float]]:
        """Run PPO training for ``total_timesteps`` environment steps.

        Returns a dict with training metrics history:
        ``{"pg_loss", "v_loss", "entropy", "mean_reward", "update"}``.
        """
        cfg = self.cfg
        batch_size = cfg.num_steps * cfg.num_envs
        minibatch_size = batch_size // cfg.num_minibatches
        num_updates = total_timesteps // batch_size

        metrics_history: dict[str, list[float]] = {
            "pg_loss": [],
            "v_loss": [],
            "entropy": [],
            "mean_reward": [],
            "update": [],
        }

        obs, _ = self.env.reset()
        global_step = 0
        t_start = time.perf_counter()

        for update in range(1, num_updates + 1):
            # --- Collect rollout ---
            buf, obs, ep_rewards = self._collect_rollout(obs)
            global_step += batch_size

            # --- Normalise advantages ---
            adv = buf.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --- Flatten for minibatch sampling ---
            flat_obs, flat_acts, flat_lp, flat_adv, flat_ret, flat_vals = flatten_rollout(
                dataclasses.replace(buf, advantages=adv)
            )

            # --- PPO epochs ---
            key = jax.random.PRNGKey(update)
            pg_losses, v_losses, entropies = [], [], []

            for _ in range(cfg.num_epochs):
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, batch_size)
                flat_obs_p = flat_obs[perm]
                flat_acts_p = flat_acts[perm]
                flat_lp_p = flat_lp[perm]
                flat_adv_p = flat_adv[perm]
                flat_ret_p = flat_ret[perm]
                flat_vals_p = flat_vals[perm]

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_obs = flat_obs_p[start:end]
                    mb_acts = flat_acts_p[start:end]
                    mb_lp = flat_lp_p[start:end]
                    mb_adv = flat_adv_p[start:end]
                    mb_ret = flat_ret_p[start:end]
                    mb_vals = flat_vals_p[start:end]

                    self._state, losses = _ppo_update_step(
                        self._state,
                        mb_obs,
                        mb_acts,
                        mb_lp,
                        mb_vals,
                        mb_adv,
                        mb_ret,
                        cfg.clip_coef,
                        cfg.ent_coef,
                        cfg.vf_coef,
                    )
                    pg_losses.append(float(losses["pg_loss"]))
                    v_losses.append(float(losses["v_loss"]))
                    entropies.append(float(losses["entropy"]))

            # --- Logging ---
            if update % cfg.log_interval == 0 or update == 1:
                mean_pg = float(np.mean(pg_losses))
                mean_v = float(np.mean(v_losses))
                mean_ent = float(np.mean(entropies))
                mean_rew = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
                elapsed = time.perf_counter() - t_start
                sps = global_step / elapsed

                metrics_history["update"].append(update)
                metrics_history["pg_loss"].append(mean_pg)
                metrics_history["v_loss"].append(mean_v)
                metrics_history["entropy"].append(mean_ent)
                metrics_history["mean_reward"].append(mean_rew)

                if self._wandb is not None:
                    self._wandb.log({
                        "train/reward_mean": mean_rew,
                        "train/pg_loss": mean_pg,
                        "train/value_loss": mean_v,
                        "train/entropy": mean_ent,
                        "train/steps_per_sec": sps,
                    }, step=global_step)

                print(
                    f"update={update:4d}  "
                    f"steps={global_step:>8d}  "
                    f"sps={sps:>7.0f}  "
                    f"reward={mean_rew:+.3f}  "
                    f"pg={mean_pg:+.4f}  "
                    f"vf={mean_v:.4f}  "
                    f"ent={mean_ent:.3f}"
                )

        if self._wandb is not None:
            self._wandb.finish()

        return metrics_history

    def save(self, path: str) -> None:
        """Save network parameters to a pickle file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.device_get(self._state.params),
                    "cfg": self.cfg,
                    "obs_dim": self._obs_dim,
                    "act_dim": self._act_dim,
                },
                f,
            )
        print(f"Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        """Load network parameters from a pickle file."""
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self._state = self._state.replace(params=ckpt["params"])
        print(f"Loaded checkpoint ← {path}")

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample actions from the current policy.

        Args:
            obs:           (num_envs, obs_dim) numpy array
            deterministic: if True, return mean action (no sampling)

        Returns:
            actions: (num_envs, act_dim) numpy array
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        if deterministic:
            mean, _, _ = self._jit_apply({"params": self._state.params}, obs_jax)
            return np.array(mean)
        key = jax.random.PRNGKey(int(time.time() * 1e9) % (2**31))
        actions, _, _, _ = self._jit_act(self._state.params, obs_jax, key)
        return np.array(actions)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_rollout(
        self, obs: np.ndarray
    ) -> tuple[RolloutBuffer, np.ndarray, list[float]]:
        """Step the env ``num_steps`` times and collect trajectory data."""
        cfg = self.cfg
        T = cfg.num_steps
        N = cfg.num_envs

        obs_buf = np.zeros((T, N, self._obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, N, self._act_dim), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=bool)
        val_buf = np.zeros((T, N), dtype=np.float32)
        lp_buf = np.zeros((T, N), dtype=np.float32)

        ep_rewards: list[float] = []
        ep_reward_acc = np.zeros(N, dtype=np.float32)

        key = jax.random.PRNGKey(int(time.time() * 1e9) % (2**31))

        for t in range(T):
            obs_jax = jnp.asarray(obs, dtype=jnp.float32)
            actions, log_probs, values, key = self._jit_act(
                self._state.params, obs_jax, key
            )

            obs_buf[t] = obs
            act_buf[t] = np.array(actions)
            val_buf[t] = np.array(values)
            lp_buf[t] = np.array(log_probs)

            obs, rewards, terminated, truncated, _ = self.env.step(np.array(actions))
            dones = terminated | truncated

            rew_buf[t] = rewards
            done_buf[t] = dones
            ep_reward_acc += rewards

            for i, d in enumerate(dones):
                if d:
                    ep_rewards.append(float(ep_reward_acc[i]))
                    ep_reward_acc[i] = 0.0

        # Bootstrap last value
        last_obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        _, _, last_value, _ = self._jit_act(
            self._state.params, last_obs_jax, key
        )
        last_value_np = np.array(last_value)

        # GAE
        advantages, returns = compute_gae(
            jnp.array(rew_buf),
            jnp.array(val_buf),
            jnp.array(done_buf),
            jnp.array(last_value_np),
            cfg.gamma,
            cfg.gae_lambda,
        )

        buf = RolloutBuffer(
            obs=jnp.array(obs_buf),
            actions=jnp.array(act_buf),
            rewards=jnp.array(rew_buf),
            dones=jnp.array(done_buf),
            values=jnp.array(val_buf),
            log_probs=jnp.array(lp_buf),
            advantages=advantages,
            returns=returns,
        )
        return buf, obs, ep_rewards
