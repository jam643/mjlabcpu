"""Joint action terms — NOT JIT-compiled (write to mjData.ctrl C arrays)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from mjlabcpu.managers.action_manager import ActionTerm, ActionTermCfg
from mjlabcpu.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv


class JointPositionAction(ActionTerm):
    """Writes joint position targets to ``mjData.ctrl``.

    Action is ``(num_envs, n_actuators)`` array of target joint positions.

    Expected ``cfg.params`` keys:
        - ``entity_cfg`` (:class:`SceneEntityCfg`): entity to actuate.
        - ``scale`` (float, default 1.0): multiplier on action.
        - ``use_default_offset`` (bool, default True): if True, target = default + scale * action.

    NOT JIT-compiled — writes to ``mjData.ctrl`` (C arrays).
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        entity_cfg: SceneEntityCfg = params.get("entity_cfg", SceneEntityCfg(name="robot"))
        self._resolved = entity_cfg.resolve(env.scene)
        self._scale: float = params.get("scale", 1.0)
        self._use_default_offset: bool = params.get("use_default_offset", True)

    @property
    def action_dim(self) -> int:
        return int(self._resolved.actuator_ids.shape[0])

    def process_actions(self, actions: jnp.ndarray) -> None:
        """Rescale actions and add default offset."""
        self._raw_actions = actions
        if self._use_default_offset:
            n_act = self.action_dim
            default = self._resolved.default_qpos[:n_act]
            self._processed_actions = default + self._scale * actions
        else:
            self._processed_actions = self._scale * actions

    def apply_actions(self) -> None:
        """Write processed actions to ``mjData.ctrl`` for each environment."""
        if self._processed_actions is None:
            return
        act = np.array(self._processed_actions)  # (num_envs, n_act)
        act_ids = np.array(self._resolved.actuator_ids)  # (n_act,)
        for i, data in enumerate(self._env.sim.data):
            data.ctrl[act_ids] = act[i]

    def compute_ctrl_jax(self, ctrl: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Pure-JAX ctrl update for MJX backend.

        Args:
            ctrl:    (num_envs, nu) current ctrl array.
            actions: (num_envs, n_act) raw actions for this term.

        Returns:
            Updated ctrl array with actuator columns set.
        """
        if self._use_default_offset:
            n_act = self.action_dim
            default = self._resolved.default_qpos[:n_act]
            processed = default + self._scale * actions
        else:
            processed = self._scale * actions
        return ctrl.at[:, self._resolved.actuator_ids].set(processed)


class JointVelocityAction(ActionTerm):
    """Writes joint velocity targets to ``mjData.ctrl``.

    Expected ``cfg.params`` keys:
        - ``entity_cfg`` (:class:`SceneEntityCfg`): entity to actuate.
        - ``scale`` (float, default 1.0): multiplier on action.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        entity_cfg: SceneEntityCfg = params.get("entity_cfg", SceneEntityCfg(name="robot"))
        self._resolved = entity_cfg.resolve(env.scene)
        self._scale: float = params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return int(self._resolved.actuator_ids.shape[0])

    def process_actions(self, actions: jnp.ndarray) -> None:
        """Apply scale and store. Matches JointPositionAction pattern."""
        self._raw_actions = actions
        self._processed_actions = self._scale * actions

    def apply_actions(self) -> None:
        if self._processed_actions is None:
            return
        act = np.array(self._processed_actions)  # already scaled in process_actions
        act_ids = np.array(self._resolved.actuator_ids)
        for i, data in enumerate(self._env.sim.data):
            data.ctrl[act_ids] = act[i]

    def compute_ctrl_jax(self, ctrl: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        processed = self._scale * actions
        return ctrl.at[:, self._resolved.actuator_ids].set(processed)
