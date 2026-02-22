"""Action manager — processes actions and writes them to mjData.ctrl."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from mjlabcpu.managers.manager_base import ManagerBase

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class ActionTermCfg:
    """Configuration for a single action term."""

    cls: type  # ActionTerm subclass to instantiate
    params: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to the ActionTerm constructor."""


class ActionTerm(ABC):
    """Base class for action terms.

    Action terms are **not** JIT-compiled — they write directly to ``mjData.ctrl``
    (C arrays). Each subclass implements :meth:`apply_actions` which reads from
    ``self._processed_actions`` and writes to the simulation.
    """

    def __init__(self, cfg: ActionTermCfg, env: "ManagerBasedRlEnv") -> None:
        self.cfg = cfg
        self._env = env
        self._processed_actions: jnp.ndarray | None = None
        self._raw_actions: jnp.ndarray | None = None

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action space for this term."""

    @property
    def processed_actions(self) -> jnp.ndarray:
        if self._processed_actions is None:
            raise RuntimeError("No action has been processed yet.")
        return self._processed_actions

    @property
    def raw_actions(self) -> jnp.ndarray:
        if self._raw_actions is None:
            raise RuntimeError("No action has been processed yet.")
        return self._raw_actions

    def process_actions(self, actions: jnp.ndarray) -> None:
        """Pre-process raw actions (e.g., rescaling). Store in ``_processed_actions``."""
        self._raw_actions = actions
        self._processed_actions = actions

    @abstractmethod
    def apply_actions(self) -> None:
        """Write processed actions to ``mjData.ctrl``."""

    def compute_ctrl_jax(self, ctrl: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX ctrl update. Used by :class:`MjxManagerBasedRlEnv`.

        Args:
            ctrl:    (num_envs, nu) current ctrl array — update in-place (functionally).
            actions: (num_envs, action_dim) actions for this term.

        Returns:
            Updated ctrl array.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_ctrl_jax(). "
            "Required for MjxManagerBasedRlEnv."
        )


class ActionManager(ManagerBase):
    """Manages one or more :class:`ActionTerm` instances.

    Aggregates all term action dimensions into a single flat action vector.
    """

    def __init__(
        self,
        cfg: dict[str, ActionTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._terms: dict[str, ActionTerm] = {}
        self._build_terms()

    def _build_terms(self) -> None:
        for name, term_cfg in self._cfg.items():
            self._terms[name] = term_cfg.cls(term_cfg, self._env)

    @property
    def action_dim(self) -> int:
        return sum(t.action_dim for t in self._terms.values())

    @property
    def action_space(self):
        """Gymnasium Box action space."""
        import gymnasium as gym

        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
        )

    def process_actions(self, actions: jnp.ndarray) -> None:
        """Split flat action vector and dispatch to each term."""
        offset = 0
        for term in self._terms.values():
            dim = term.action_dim
            term.process_actions(actions[:, offset : offset + dim])
            offset += dim

    def apply_actions(self) -> None:
        """Write all term actions to mjData."""
        for term in self._terms.values():
            term.apply_actions()

    def get_processed_actions(self) -> jnp.ndarray:
        """Concatenate processed actions from all terms."""
        parts = [t.processed_actions for t in self._terms.values()]
        return jnp.concatenate(parts, axis=-1)

    def get_raw_actions(self) -> jnp.ndarray:
        """Concatenate raw actions from all terms."""
        parts = [t.raw_actions for t in self._terms.values()]
        return jnp.concatenate(parts, axis=-1)

    def compute_ctrl_jax(self, actions: jnp.ndarray, nu: int) -> jnp.ndarray:
        """Compute the full ctrl array from a flat action vector. Pure JAX.

        Used by :class:`MjxManagerBasedRlEnv` to apply actions inside JIT.

        Args:
            actions: (num_envs, total_action_dim) flat action vector.
            nu:      Number of MuJoCo actuators (model.nu).

        Returns:
            ctrl: (num_envs, nu) control array.
        """
        ctrl = jnp.zeros((actions.shape[0], nu))
        offset = 0
        for term in self._terms.values():
            dim = term.action_dim
            ctrl = term.compute_ctrl_jax(ctrl, actions[:, offset : offset + dim])
            offset += dim
        return ctrl

    def reset(self, env_ids: list[int] | None = None) -> None:
        pass
