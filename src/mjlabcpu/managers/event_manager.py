"""Event manager — handles discrete events like resets."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

from mjlabcpu.managers.manager_base import ManagerBase

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclasses.dataclass
class EventTermCfg:
    """Configuration for a single event term."""

    func: Callable[..., None]
    """Function to call. Signature: ``(env, env_ids, **params)``."""
    mode: str = "reset"
    """When to fire this event. Currently only 'reset' is supported."""
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Extra keyword arguments passed to the event function."""


class EventManager(ManagerBase):
    """Manages event functions (e.g., randomisation on reset).

    Event functions are **not** JIT-compiled — they directly modify ``mjData``
    (C arrays) and environment state.
    """

    def __init__(
        self,
        cfg: dict[str, EventTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(env)
        self._cfg = cfg

    def apply_reset(self, env_ids: list[int]) -> None:
        """Fire all 'reset' mode events for the given env IDs."""
        for term_name, term_cfg in self._cfg.items():
            if term_cfg.mode == "reset":
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    def apply_interval(self, env_ids: list[int]) -> None:
        """Fire all 'interval' mode events for the given env IDs."""
        for term_name, term_cfg in self._cfg.items():
            if term_cfg.mode == "interval":
                term_cfg.func(self._env, env_ids, **term_cfg.params)
