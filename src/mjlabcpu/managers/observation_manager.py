"""Observation manager with JIT-compiled compute pipeline."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from mjlabcpu.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlabcpu.managers.scene_entity_cfg import SceneEntityCfg
from mjlabcpu.sim.sim_state import SimState

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for a single observation term."""

    clip: tuple[float, float] | None = None
    """Optional clipping range (min, max)."""
    scale: float = 1.0
    """Scalar multiplier applied after the term function."""


@dataclasses.dataclass
class ObservationGroupCfg:
    """Configuration for a group of observation terms (e.g., 'policy', 'critic')."""

    terms: dict[str, ObservationTermCfg] = dataclasses.field(default_factory=dict)
    concatenate: bool = True
    """If True, concatenate all terms into a single array."""


class ObservationManager(ManagerBase):
    """Computes observations using a JIT-compiled pipeline.

    Each observation group is compiled separately. The first call to
    ``compute()`` triggers JAX tracing; subsequent calls use the compiled
    XLA computation.
    """

    def __init__(
        self,
        cfg: dict[str, ObservationGroupCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._jit_fns: dict[str, Callable[[SimState], jnp.ndarray]] = {}
        self._group_dims: dict[str, int] = {}
        self._raw_term_fns: dict[str, list[tuple[str, Callable]]] = {}
        self._prepare()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Resolve static params and JIT-compile each observation group."""
        for group_name, group_cfg in self._cfg.items():
            term_fns: list[Callable[[SimState], jnp.ndarray]] = []
            term_names: list[str] = []

            for term_name, term_cfg in group_cfg.terms.items():
                # Resolve SceneEntityCfg params to concrete arrays
                resolved_params = _resolve_params(term_cfg.params, self._env.scene)

                # Build the term partial function
                fn = functools.partial(term_cfg.func, **resolved_params)

                # Apply scale / clip wrappers
                scale = term_cfg.scale
                clip = term_cfg.clip
                fn = _wrap_term(fn, scale, clip)
                term_fns.append(fn)
                term_names.append(term_name)

            # Store raw (non-JIT) term functions for monitoring
            self._raw_term_fns[group_name] = list(zip(term_names, term_fns))

            # Capture term_fns in closure for JIT
            _fns = term_fns

            def _compute_group(state: SimState, fns: list = _fns) -> jnp.ndarray:
                parts = [f(state) for f in fns]
                if not parts:
                    return jnp.zeros((state.qpos.shape[0], 0))
                return jnp.concatenate(parts, axis=-1)

            self._jit_fns[group_name] = jax.jit(_compute_group)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self, state: SimState) -> dict[str, jnp.ndarray]:
        """Compute all observation groups. Returns dict of group_name → array."""
        return {name: fn(state) for name, fn in self._jit_fns.items()}

    def compute_group(self, group_name: str, state: SimState) -> jnp.ndarray:
        """Compute a single observation group."""
        return self._jit_fns[group_name](state)

    def compute_terms(self, state: SimState) -> dict[str, jnp.ndarray]:
        """Return {term_name: array} for all terms. Non-JIT — for monitoring only."""
        result = {}
        for _group, name_fn_pairs in self._raw_term_fns.items():
            for term_name, fn in name_fn_pairs:
                result[term_name] = fn(state)
        return result

    @property
    def group_names(self) -> list[str]:
        return list(self._cfg.keys())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _wrap_term(
    fn: Callable[[SimState], jnp.ndarray],
    scale: float,
    clip: tuple[float, float] | None,
) -> Callable[[SimState], jnp.ndarray]:
    """Wrap a term function with optional scale and clip."""
    if scale != 1.0 or clip is not None:

        def wrapped(state: SimState) -> jnp.ndarray:
            out = fn(state)
            if scale != 1.0:
                out = out * scale
            if clip is not None:
                out = jnp.clip(out, clip[0], clip[1])
            return out

        return wrapped
    return fn


def _resolve_params(params: dict[str, Any], scene: Any) -> dict[str, Any]:
    """Replace SceneEntityCfg values with their resolved counterparts."""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, SceneEntityCfg):
            resolved[k] = v.resolve(scene)
        else:
            resolved[k] = v
    return resolved
