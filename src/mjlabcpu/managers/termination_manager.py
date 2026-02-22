"""Termination manager with JIT-compiled compute pipeline."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from mjlabcpu.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlabcpu.managers.scene_entity_cfg import SceneEntityCfg
from mjlabcpu.sim.sim_state import SimState

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a single termination term."""

    time_out: bool = False
    """If True, this term counts as a timeout (truncation) rather than failure."""


class TerminationManager(ManagerBase):
    """Computes episode termination using a JIT-compiled pipeline.

    Returns boolean arrays for ``done`` (any termination) and ``truncated``
    (timeout-only termination), both of shape ``(num_envs,)``.
    """

    def __init__(
        self,
        cfg: dict[str, TerminationTermCfg],
        env: ManagerBasedRlEnv,
    ) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._jit_compute: Callable[
            [SimState], tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]
        ]
        self._prepare()

    def _prepare(self) -> None:
        term_fns: list[Callable[[SimState], jnp.ndarray]] = []
        is_timeout: list[bool] = []
        term_names: list[str] = []

        for term_name, term_cfg in self._cfg.items():
            resolved_params = _resolve_params(term_cfg.params, self._env.scene)
            fn = functools.partial(term_cfg.func, **resolved_params)
            term_fns.append(fn)
            is_timeout.append(term_cfg.time_out)
            term_names.append(term_name)

        _fns = term_fns
        _timeouts = is_timeout
        _names = term_names

        def _compute(
            state: SimState,
            fns: list = _fns,
            timeouts: list = _timeouts,
        ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
            num_envs = state.qpos.shape[0]
            done = jnp.zeros(num_envs, dtype=jnp.bool_)
            truncated = jnp.zeros(num_envs, dtype=jnp.bool_)
            terms_out = {}

            for fn, is_to, n in zip(fns, timeouts, _names, strict=True):
                v = fn(state)  # (num_envs,) bool
                terms_out[n] = v
                done = done | v
                if is_to:
                    truncated = truncated | v

            return done, truncated, terms_out

        self._jit_compute = jax.jit(_compute)

    def compute(self, state: SimState) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute termination signals.

        Returns:
            done: (num_envs,) bool — any termination (failure or timeout).
            truncated: (num_envs,) bool — timeout-only termination.
            terms: dict of term_name → (num_envs,) bool.
        """
        return self._jit_compute(state)


def _resolve_params(params: dict[str, Any], scene: Any) -> dict[str, Any]:
    resolved = {}
    for k, v in params.items():
        if isinstance(v, SceneEntityCfg):
            resolved[k] = v.resolve(scene)
        else:
            resolved[k] = v
    return resolved
