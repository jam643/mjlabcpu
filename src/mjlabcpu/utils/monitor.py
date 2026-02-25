"""Real-time episode monitor using rerun-sdk.

Logs reward terms, observation terms, actions, and termination signals
to a live rerun dashboard.

Timeline design
---------------
* ``total_step`` — monotonically increasing, used for all per-step scalars.
  At episode boundaries a ``nan`` is inserted so the line plot shows a
  visual gap/break between episodes (the standard rerun idiom; ``rr.Clear``
  only punches a hole at one point and cannot retroactively remove earlier
  data from range queries).
* ``episode`` — increments once per episode, used only for
  ``episode/return`` so you get a clean learning-curve plot.

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
        self._total_step = 0
        self._episode = 0
        self._ep_return = 0.0
        self._scalar_paths: list[str] = []  # ordered list for gap logging
        self._rr = rr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scalar(self, path: str, value: float) -> None:
        """Log a scalar on the ``total_step`` timeline and track the path."""
        self._rr.log(path, self._rr.Scalars(value))
        if path not in self._scalar_paths:
            self._scalar_paths.append(path)

    def _log_gap(self) -> None:
        """Insert a nan at the current ``total_step`` for all tracked scalar paths.

        This creates a visual line-break between episodes without removing
        historical data (which rerun does not support after the fact).
        """
        rr = self._rr
        rr.set_time("total_step", sequence=self._total_step)
        for path in self._scalar_paths:
            rr.log(path, rr.Scalars(float("nan")))
        self._total_step += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        rr.set_time("total_step", sequence=self._total_step)

        step_reward = float(rewards[i])
        self._ep_return += step_reward

        # --- Total reward ---
        self._scalar("reward/total", step_reward)

        # --- Per-term rewards ---
        for name, vals in info.get("reward_terms", {}).items():
            self._scalar(f"reward/{name}", float(vals[i]))

        # --- Observations (per-term) ---
        for name, arr in obs_terms.items():
            a = np.array(arr[i])
            if a.ndim == 0 or a.size == 1:
                self._scalar(f"obs/{name}", float(a.flat[0]))
            else:
                for d, v in enumerate(a.flat):
                    self._scalar(f"obs/{name}/{d}", float(v))

        # --- Actions (per-dim, grouped by action term name) ---
        act = np.array(action[i])
        offset = 0
        for term_name, term in self._env._action_manager._terms.items():
            dim = term.action_dim
            for d in range(dim):
                self._scalar(f"action/{term_name}/{d}", float(act[offset + d]))
            offset += dim

        # --- Termination signals ---
        for name, vals in info.get("termination_terms", {}).items():
            self._scalar(f"termination/{name}", float(bool(vals[i])))

        self._total_step += 1

        # --- Episode boundary ---
        done = bool(terminated[i]) or bool(truncated[i])
        if done:
            # Cumulative episode return on its own timeline
            rr.set_time("episode", sequence=self._episode)
            rr.log("episode/return", rr.Scalars(self._ep_return))

            reason = "truncated" if truncated[i] else "terminated"
            rr.log(
                "episode/reset",
                rr.TextLog(
                    f"Episode {self._episode} ended ({reason})  return={self._ep_return:+.2f}"
                ),
            )

            # Insert nan gap so line plots show a break between episodes
            self._log_gap()

            self._episode += 1
            self._ep_return = 0.0
