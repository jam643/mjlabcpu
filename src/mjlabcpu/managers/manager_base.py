"""Base classes for all managers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class ManagerTermBaseCfg:
    """Configuration for a single manager term."""

    func: Callable[..., Any]
    """The term function. Signature depends on the manager type."""
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Additional keyword arguments passed to ``func`` at construction time."""


class ManagerBase:
    """Abstract base class for all managers.

    Each manager is tied to a :class:`ManagerBasedRlEnv` and can reference
    the scene, sim, and other managers through the env.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self._env = env

    @property
    def env(self) -> ManagerBasedRlEnv:
        return self._env

    def reset(self, env_ids: list[int] | None = None) -> None:  # noqa: B027
        """Reset manager state for the given env IDs (or all if None)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
