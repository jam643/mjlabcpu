"""Entity — represents a single articulated body or asset in the scene."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import mujoco
import numpy as np

if TYPE_CHECKING:
    from mjlabcpu.scene.scene import Scene


@dataclasses.dataclass
class EntityCfg:
    """Configuration for spawning an entity into the scene.

    Attributes:
        prim_path: Unique scene identifier (used as name prefix in MjSpec).
        spawn: Path to the MJCF XML file describing this entity.
        init_state: Optional initial state overrides.
    """

    prim_path: str
    """Unique identifier / spawn path (becomes the MjSpec prefix)."""
    spawn: str | None = None
    """Path to the MJCF XML for this entity. If None, entity is defined inline."""
    init_state: "InitState | None" = None
    """Initial state (joint positions, root pose, etc.)."""


@dataclasses.dataclass
class InitState:
    """Initial state for an entity at reset."""

    joint_pos: dict[str, float] = dataclasses.field(default_factory=dict)
    """Mapping of joint name → default position."""
    joint_vel: dict[str, float] = dataclasses.field(default_factory=dict)
    """Mapping of joint name → default velocity."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Root body position."""
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Root body quaternion (wxyz)."""
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Root body linear velocity."""
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Root body angular velocity."""


@dataclasses.dataclass
class EntityIndexing:
    """Pre-computed integer index arrays for fast JAX slicing into SimState.

    All arrays are ``jnp.ndarray`` of dtype ``int32`` so they can be used
    directly as JAX indices inside JIT-compiled functions.
    """

    # Body indices
    body_ids: jnp.ndarray  # shape (nbody,) — model body indices for this entity
    root_body_id: int  # scalar — model index of the root body

    # Joint/DOF indices
    joint_ids: jnp.ndarray  # shape (njoint,) — model joint indices
    qpos_addrs: jnp.ndarray  # shape (nq,) — positions in qpos array
    qvel_addrs: jnp.ndarray  # shape (nv,) — positions in qvel array

    # Actuator indices
    actuator_ids: jnp.ndarray  # shape (nu,)

    # Default state (for computing relative obs)
    default_qpos: jnp.ndarray  # shape (nq,)
    default_qvel: jnp.ndarray  # shape (nv,)


class Entity:
    """A registered entity within a :class:`~mjlabcpu.scene.scene.Scene`.

    After the scene is built, each entity holds its :class:`EntityIndexing`
    with pre-computed JAX index arrays derived from the compiled ``MjModel``.
    """

    def __init__(self, cfg: EntityCfg, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self._indexing: EntityIndexing | None = None

    # ------------------------------------------------------------------
    # Index resolution (called by Scene after model is compiled)
    # ------------------------------------------------------------------

    def resolve(self, model: mujoco.MjModel, prefix: str) -> None:
        """Build :class:`EntityIndexing` from the compiled model.

        Args:
            model: The compiled ``mujoco.MjModel``.
            prefix: Name prefix (e.g. ``"robot/"``).
        """
        # Collect body ids with this prefix
        body_ids = []
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if bname.startswith(prefix) or bname == prefix.rstrip("/"):
                body_ids.append(i)

        if not body_ids:
            # Fallback: body 0 is the world body; try without prefix for simple models
            for i in range(model.nbody):
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                if bname and bname != "world":
                    body_ids.append(i)

        root_body_id = body_ids[0] if body_ids else 0

        # Collect joint ids and their qpos/qvel addresses
        joint_ids = []
        qpos_addrs = []
        qvel_addrs = []
        for j in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if jname.startswith(prefix) or (not prefix):
                joint_ids.append(j)
                # qpos address
                addr = model.jnt_qposadr[j]
                nq_j = _joint_nq(model.jnt_type[j])
                qpos_addrs.extend(range(addr, addr + nq_j))
                # qvel address
                vaddr = model.jnt_dofadr[j]
                nv_j = _joint_nv(model.jnt_type[j])
                qvel_addrs.extend(range(vaddr, vaddr + nv_j))

        # Collect actuator ids with this prefix
        actuator_ids = []
        for a in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if aname.startswith(prefix) or (not prefix and model.nu > 0):
                actuator_ids.append(a)

        # Build default qpos/qvel (from model key_qpos if available, else qpos0)
        default_qpos_full = np.zeros(model.nq)
        if model.nkey > 0:
            default_qpos_full = model.key_qpos[0].copy()
        else:
            # Use model.qpos0 which stores default joint positions
            default_qpos_full = np.zeros(model.nq)
            for j in range(model.njnt):
                addr = model.jnt_qposadr[j]
                nq_j = _joint_nq(model.jnt_type[j])
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    # Free joint: default to identity orientation
                    default_qpos_full[addr : addr + 3] = 0.0
                    default_qpos_full[addr + 3 : addr + 7] = [1, 0, 0, 0]
                else:
                    default_qpos_full[addr : addr + nq_j] = 0.0

        self._indexing = EntityIndexing(
            body_ids=jnp.array(body_ids, dtype=jnp.int32),
            root_body_id=root_body_id,
            joint_ids=jnp.array(joint_ids, dtype=jnp.int32),
            qpos_addrs=jnp.array(qpos_addrs, dtype=jnp.int32),
            qvel_addrs=jnp.array(qvel_addrs, dtype=jnp.int32),
            actuator_ids=jnp.array(actuator_ids, dtype=jnp.int32),
            default_qpos=jnp.array(
                default_qpos_full[qpos_addrs] if qpos_addrs else np.zeros(0), dtype=jnp.float32
            ),
            default_qvel=jnp.array(
                np.zeros(len(qvel_addrs)), dtype=jnp.float32
            ),
        )

    @property
    def indexing(self) -> EntityIndexing:
        if self._indexing is None:
            raise RuntimeError(f"Entity '{self.name}' has not been resolved against the model.")
        return self._indexing

    # Convenience passthrough properties
    @property
    def body_ids(self) -> jnp.ndarray:
        return self.indexing.body_ids

    @property
    def root_body_id(self) -> int:
        return self.indexing.root_body_id

    @property
    def qpos_addrs(self) -> jnp.ndarray:
        return self.indexing.qpos_addrs

    @property
    def qvel_addrs(self) -> jnp.ndarray:
        return self.indexing.qvel_addrs

    @property
    def actuator_ids(self) -> jnp.ndarray:
        return self.indexing.actuator_ids

    @property
    def default_qpos(self) -> jnp.ndarray:
        return self.indexing.default_qpos

    @property
    def default_qvel(self) -> jnp.ndarray:
        return self.indexing.default_qvel


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _joint_nq(jnt_type: int) -> int:
    """Number of qpos entries for a joint type."""
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1  # mjJNT_HINGE or mjJNT_SLIDE


def _joint_nv(jnt_type: int) -> int:
    """Number of qvel entries for a joint type."""
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1
