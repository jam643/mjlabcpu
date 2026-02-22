"""Reward manager with JIT-compiled compute pipeline."""

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
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a single reward term."""

    weight: float = 1.0
    """Scalar weight applied to this term's output."""


class RewardManager(ManagerBase):
    """Computes scalar rewards using a JIT-compiled pipeline.

    All terms are summed (weighted) into a single reward vector of shape
    ``(num_envs,)``.
    """

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        env: ManagerBasedRlEnv,
    ) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._jit_compute: Callable[[SimState], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]
        self._prepare()

    def _prepare(self) -> None:
        """Resolve static params and JIT-compile the reward compute function."""
        term_fns: list[Callable[[SimState], jnp.ndarray]] = []
        weights: list[float] = []
        term_names: list[str] = []

        for term_name, term_cfg in self._cfg.items():
            resolved_params = _resolve_params(term_cfg.params, self._env.scene)
            fn = functools.partial(term_cfg.func, **resolved_params)
            term_fns.append(fn)
            weights.append(term_cfg.weight)
            term_names.append(term_name)

        _fns = term_fns
        _weights = weights
        _names = term_names

        def _compute(
            state: SimState,
            fns: list = _fns,
            ws: list = _weights,
        ) -> tuple[jnp.ndarray, dict]:
            terms_out = {}
            total = None
            for fn, w, n in zip(fns, ws, _names, strict=True):
                v = fn(state)  # (num_envs,)
                terms_out[n] = v
                weighted = v * w
                total = weighted if total is None else total + weighted
            if total is None:
                total = jnp.zeros(state.qpos.shape[0])
            return total, terms_out

        self._jit_compute = jax.jit(_compute)

    def compute(self, state: SimState) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute total reward and per-term breakdowns.

        Returns:
            total: (num_envs,) summed weighted reward.
            terms: dict of term_name → (num_envs,) individual rewards.
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
