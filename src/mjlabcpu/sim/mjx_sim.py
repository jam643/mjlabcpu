"""MJX-based parallel simulation — batched JAX pytree physics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mjlabcpu.sim.sim import SimulationCfg


class MjxSimulation:
    """MuJoCo MJX simulation backend.

    Stores the full batch of simulation states as a single batched ``mjx.Data``
    JAX pytree. Physics is stepped with ``jax.vmap`` + ``jax.lax.scan`` inside
    a JIT-compiled function (see :meth:`MjxManagerBasedRlEnv._build_jit_step`).

    Unlike :class:`~mjlabcpu.sim.sim.Simulation`, there are no C ``MjData``
    objects — all state lives in JAX arrays on the accelerator.
    """

    def __init__(
        self, model: mujoco.MjModel, num_envs: int, cfg: SimulationCfg
    ) -> None:
        # Apply config overrides to the model (same as CPU Simulation)
        model.opt.timestep = cfg.dt
        model.opt.gravity[:] = cfg.gravity
        model.opt.integrator = cfg.mujoco.integrator
        model.opt.solver = cfg.mujoco.solver
        model.opt.iterations = cfg.mujoco.iterations
        model.opt.ls_iterations = cfg.mujoco.ls_iterations

        self.model = model
        self._mjx_model = mjx.put_model(model)
        self._num_envs = num_envs
        self._cfg = cfg

        # Build single-env initial state (reset + forward kinematics)
        _data = mujoco.MjData(model)
        mujoco.mj_resetData(model, _data)
        mujoco.mj_forward(model, _data)
        self._init_mjx_data = mjx.put_data(model, _data)

        # Replicate to (num_envs, ...) batch
        self._mjx_data = self._batch(self._init_mjx_data)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mjx_model(self):
        """The ``mjx.Model`` (JAX-compatible model description)."""
        return self._mjx_model

    @property
    def mjx_data(self):
        """Batched ``mjx.Data`` with shape ``(num_envs, ...)`` on each leaf."""
        return self._mjx_data

    @mjx_data.setter
    def mjx_data(self, value) -> None:
        self._mjx_data = value

    @property
    def init_mjx_data(self):
        """Single-env initial ``mjx.Data`` used as the reset template."""
        return self._init_mjx_data

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def data(self):
        raise AttributeError(
            "MjxSimulation has no C MjData instances (.data). "
            "Use .mjx_data (batched JAX pytree) instead."
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Reset all environments to the model's default state."""
        self._mjx_data = self._batch(self._init_mjx_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch(self, single_data) -> object:
        """Stack a single-env ``mjx.Data`` into a ``(num_envs, ...)`` batch."""
        return jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * self._num_envs), single_data
        )

    def close(self) -> None:
        pass
