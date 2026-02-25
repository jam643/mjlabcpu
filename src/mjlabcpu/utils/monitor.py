"""Real-time episode monitor using rerun-sdk.

Logs reward terms, observation terms, actions, and termination signals
to a live rerun dashboard. Uses ``step_in_episode`` as the primary timeline
so the plot naturally shows only the current episode; old episodes remain
accessible via the time scrubber.

Usage::

    from mjlabcpu.utils.monitor import EnvMonitor
    monitor = EnvMonitor(env)
    # ... in your step loop:
    obs_terms = env._obs_manager.compute_terms(env._make_dummy_state())
    monitor.log_step(obs_terms, rewards, terminated, truncated, info, action)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


class EnvMonitor:
    """Logs RL step data to a live rerun dashboard.

    Args:
        env:     The :class:`ManagerBasedRlEnv` instance to monitor.
        env_idx: Which environment index to log (default: 0).
        app_id:  rerun application ID (appears in the viewer title bar).
    """

    def __init__(self, env: ManagerBasedRlEnv, env_idx: int = 0, app_id: str = "mjlabcpu") -> None:
        try:
            import rerun as rr
        except ImportError as e:
            raise ImportError("rerun-sdk is not installed (run: uv sync)") from e

        rr.init(app_id, spawn=True)
        self._env = env
        self._env_idx = env_idx
        self._step_in_ep = 0
        self._total_step = 0
        self._episode = 0
        self._ep_return = 0.0
        self._rr = rr

    def log_step(
        self,
        obs_terms: dict,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict,
        action: np.ndarray,
    ) -> None:
        """Log one RL step to rerun.

        Args:
            obs_terms:  Dict of term_name → array from ``obs_manager.compute_terms()``.
            rewards:    (num_envs,) total reward array.
            terminated: (num_envs,) bool — failure termination.
            truncated:  (num_envs,) bool — timeout truncation.
            info:       Dict with ``reward_terms`` and ``termination_terms`` keys.
            action:     (num_envs, action_dim) action array applied this step.
        """
        rr = self._rr
        i = self._env_idx

        rr.set_time("step_in_episode", sequence=self._step_in_ep)
        rr.set_time("total_step", sequence=self._total_step)

        step_reward = float(rewards[i])
        self._ep_return += step_reward

        # --- Total reward ---
        rr.log("reward/total", rr.Scalars(step_reward))

        # --- Per-term rewards ---
        for name, vals in info.get("reward_terms", {}).items():
            rr.log(f"reward/{name}", rr.Scalars(float(vals[i])))

        # --- Observations (per-term) ---
        for name, arr in obs_terms.items():
            a = np.array(arr[i])
            if a.ndim == 0 or a.size == 1:
                rr.log(f"obs/{name}", rr.Scalars(float(a.flat[0])))
            else:
                for d, v in enumerate(a.flat):
                    rr.log(f"obs/{name}/{d}", rr.Scalars(float(v)))

        # --- Actions (per-dim, grouped by action term name) ---
        act = np.array(action[i])
        offset = 0
        for term_name, term in self._env._action_manager._terms.items():
            dim = term.action_dim
            for d in range(dim):
                rr.log(f"action/{term_name}/{d}", rr.Scalars(float(act[offset + d])))
            offset += dim

        # --- Termination signals ---
        for name, vals in info.get("termination_terms", {}).items():
            rr.log(f"termination/{name}", rr.Scalars(float(bool(vals[i]))))

        self._step_in_ep += 1
        self._total_step += 1

        # --- Episode boundary ---
        done = bool(terminated[i]) or bool(truncated[i])
        if done:
            # --- Cumulative episode return (on the episode timeline) ---
            rr.set_time("episode", sequence=self._episode)
            rr.log("episode/return", rr.Scalars(self._ep_return))

            # --- Clear per-step plots so the new episode starts fresh ---
            # Log clears at total_step N (one past the last data point) so they
            # appear right after the episode's final step in the scrubber.
            rr.set_time("total_step", sequence=self._total_step)
            rr.set_time("step_in_episode", sequence=0)
            for path in ["reward", "obs", "action", "termination"]:
                rr.log(path, rr.Clear(recursive=True))

            reason = "truncated" if truncated[i] else "terminated"
            rr.log(
                "episode/reset",
                rr.TextLog(
                    f"Episode {self._episode} ended ({reason}) after {self._step_in_ep} steps"
                    f"  return={self._ep_return:+.2f}"
                ),
            )
            self._step_in_ep = 0
            self._episode += 1
            self._ep_return = 0.0
