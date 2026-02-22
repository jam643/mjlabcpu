"""MuJoCo simulation wrapper with CPU-parallel physics via ThreadPoolExecutor."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed

import mujoco
import numpy as np


@dataclasses.dataclass
class MujocoCfg:
    """Low-level MuJoCo solver configuration."""

    timestep: float = 0.002
    """Physics timestep in seconds."""
    integrator: int = mujoco.mjtIntegrator.mjINT_EULER
    """MuJoCo integrator type."""
    solver: int = mujoco.mjtSolver.mjSOL_NEWTON
    """MuJoCo constraint solver type."""
    iterations: int = 4
    """Number of solver iterations."""
    ls_iterations: int = 8
    """Number of line-search iterations."""


@dataclasses.dataclass
class SimulationCfg:
    """Simulation configuration."""

    dt: float = 0.002
    """Physics timestep in seconds. Overrides MujocoCfg.timestep if set."""
    render_dt: float = 1.0 / 60.0
    """Render timestep (used by viewer if attached)."""
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector."""
    mujoco: MujocoCfg = dataclasses.field(default_factory=MujocoCfg)
    """MuJoCo-specific settings."""
    max_thread_workers: int | None = None
    """Max threads for ThreadPoolExecutor. None = number of CPUs."""


class Simulation:
    """Wraps a ``mujoco.MjModel`` and ``num_envs`` independent ``mujoco.MjData`` instances.

    Physics is stepped in parallel using a ``ThreadPoolExecutor``. The GIL is
    released inside ``mj_step`` (a C call), so true parallelism is achieved.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        num_envs: int,
        cfg: SimulationCfg | None = None,
    ) -> None:
        self.cfg = cfg or SimulationCfg()
        self.num_envs = num_envs

        # Apply configuration overrides to model
        self.model = model
        self.model.opt.timestep = self.cfg.dt
        self.model.opt.gravity[:] = self.cfg.gravity
        self.model.opt.integrator = self.cfg.mujoco.integrator
        self.model.opt.solver = self.cfg.mujoco.solver
        self.model.opt.iterations = self.cfg.mujoco.iterations
        self.model.opt.ls_iterations = self.cfg.mujoco.ls_iterations

        # One MjData per environment
        self.data: list[mujoco.MjData] = [mujoco.MjData(model) for _ in range(num_envs)]

        # Thread pool for parallel physics
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.max_thread_workers)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Step all environments in parallel using ThreadPoolExecutor.

        ``mj_step`` releases the GIL (it is a C function), enabling true
        parallel execution across CPU cores.
        """
        model = self.model
        futures = {
            self._executor.submit(mujoco.mj_step, model, d): i for i, d in enumerate(self.data)
        }
        # Wait for all to finish and propagate exceptions
        for future in as_completed(futures):
            future.result()

    def forward(self) -> None:
        """Run mj_forward (kinematics + sensors, no dynamics) on all envs in parallel."""
        model = self.model
        futures = {
            self._executor.submit(mujoco.mj_forward, model, d): i for i, d in enumerate(self.data)
        }
        for future in as_completed(futures):
            future.result()

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def reset_env(self, env_id: int) -> None:
        """Reset a single environment to the model's keyframe 0 defaults."""
        mujoco.mj_resetData(self.model, self.data[env_id])
        mujoco.mj_forward(self.model, self.data[env_id])

    def reset_envs(self, env_ids: list[int] | np.ndarray) -> None:
        """Reset multiple environments."""
        for i in env_ids:
            self.reset_env(int(i))

    def reset_all(self) -> None:
        """Reset all environments."""
        for d in self.data:
            mujoco.mj_resetData(self.model, d)
            mujoco.mj_forward(self.model, d)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    @property
    def nbody(self) -> int:
        return self.model.nbody

    @property
    def nsensordata(self) -> int:
        return self.model.nsensordata

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
