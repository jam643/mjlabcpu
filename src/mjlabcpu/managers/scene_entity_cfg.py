"""SceneEntityCfg — resolves entity names to JAX index arrays at init time."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from mjlabcpu.scene.scene import Scene


@dataclasses.dataclass
class SceneEntityCfg:
    """Reference to an entity in the scene with optional joint/body name filtering.

    Used in :class:`~mjlabcpu.managers.ManagerTermBaseCfg` ``params`` to
    resolve entity index arrays at init time (before JIT compilation).

    Example::

        cfg = SceneEntityCfg(name="robot", joint_names=["LF_HAA", "LF_HFE"])
    """

    name: str
    """Name of the entity as registered in :class:`~mjlabcpu.scene.scene.SceneCfg`."""
    joint_names: list[str] | None = None
    """Whitelist of joint names to include. If None, all joints are included."""
    body_names: list[str] | None = None
    """Whitelist of body names to include. If None, all bodies are included."""

    def resolve(self, scene: Scene) -> ResolvedEntityCfg:
        """Resolve names to JAX integer index arrays from the compiled model.

        Returns a :class:`ResolvedEntityCfg` with concrete ``jnp.ndarray`` indices.
        """
        import mujoco

        entity = scene[self.name]
        model = scene.model
        prefix = f"{self.name}/"

        # --- joint/qpos/qvel resolution ---
        if self.joint_names is not None:
            qpos_addrs = []
            qvel_addrs = []
            joint_ids = []
            for jname in self.joint_names:
                # Try with prefix first, then without
                full_name = f"{prefix}{jname}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
                if jid == -1:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid == -1:
                    raise ValueError(f"Joint '{jname}' not found in model (entity '{self.name}').")
                joint_ids.append(jid)
                from mjlabcpu.entity.entity import _joint_nq, _joint_nv

                addr = model.jnt_qposadr[jid]
                nq_j = _joint_nq(model.jnt_type[jid])
                qpos_addrs.extend(range(addr, addr + nq_j))
                vaddr = model.jnt_dofadr[jid]
                nv_j = _joint_nv(model.jnt_type[jid])
                qvel_addrs.extend(range(vaddr, vaddr + nv_j))

            qpos_addrs_arr = jnp.array(qpos_addrs, dtype=jnp.int32)
            qvel_addrs_arr = jnp.array(qvel_addrs, dtype=jnp.int32)
            joint_ids_arr = jnp.array(joint_ids, dtype=jnp.int32)
        else:
            qpos_addrs_arr = entity.qpos_addrs
            qvel_addrs_arr = entity.qvel_addrs
            joint_ids_arr = entity.indexing.joint_ids

        # --- body resolution ---
        if self.body_names is not None:
            body_ids = []
            for bname in self.body_names:
                full_name = f"{prefix}{bname}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_name)
                if bid == -1:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
                if bid == -1:
                    raise ValueError(f"Body '{bname}' not found in model (entity '{self.name}').")
                body_ids.append(bid)
            body_ids_arr = jnp.array(body_ids, dtype=jnp.int32)
        else:
            body_ids_arr = entity.body_ids

        # Default qpos for selected joints — O(n) lookup via dict instead of O(n²)
        default_qpos = entity.default_qpos
        if self.joint_names is not None and len(qpos_addrs) > 0:
            entity_qpos_index = {addr: i for i, addr in enumerate(entity.qpos_addrs.tolist())}
            selected_indices = [entity_qpos_index[a] for a in qpos_addrs if a in entity_qpos_index]
            if selected_indices:
                selected = jnp.array(selected_indices, dtype=jnp.int32)
                default_qpos = entity.default_qpos[selected]
            else:
                default_qpos = jnp.zeros(len(qpos_addrs), dtype=jnp.float32)

        return ResolvedEntityCfg(
            entity=entity,
            body_ids=body_ids_arr,
            root_body_id=entity.root_body_id,
            joint_ids=joint_ids_arr,
            qpos_addrs=qpos_addrs_arr,
            qvel_addrs=qvel_addrs_arr,
            default_qpos=default_qpos,
            actuator_ids=entity.actuator_ids,
        )


@dataclasses.dataclass
class ResolvedEntityCfg:
    """Resolved entity configuration with concrete JAX index arrays."""

    entity: object  # Entity — avoid circular import type
    body_ids: jnp.ndarray
    root_body_id: int
    joint_ids: jnp.ndarray
    qpos_addrs: jnp.ndarray
    qvel_addrs: jnp.ndarray
    default_qpos: jnp.ndarray
    actuator_ids: jnp.ndarray
