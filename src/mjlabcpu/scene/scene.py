"""Scene — assembles a MuJoCo model from entity MJCF files using MjSpec."""

from __future__ import annotations

import dataclasses
import os

import mujoco

from mjlabcpu.entity.entity import Entity, EntityCfg


@dataclasses.dataclass
class SceneCfg:
    """Configuration for scene construction.

    Attributes:
        num_envs: Number of parallel environments. Currently all envs share one
            MjModel but have independent MjData.
        ground_plane: Whether to add a ground plane geom.
        ground_plane_cfg: Kwargs passed to the ground plane geom.
        entities: Mapping of name → :class:`EntityCfg`.
        light: Whether to add a default headlight.
    """

    num_envs: int = 1
    ground_plane: bool = True
    light: bool = True
    entities: dict[str, EntityCfg] = dataclasses.field(default_factory=dict)


class Scene:
    """Assembles a ``mujoco.MjModel`` from entity MJCF files using ``mujoco.MjSpec``.

    Usage::

        cfg = SceneCfg(
            num_envs=4,
            entities={"robot": EntityCfg(prim_path="robot", spawn="robot.xml")},
        )
        scene = Scene(cfg)
        model = scene.model
    """

    def __init__(self, cfg: SceneCfg) -> None:
        self.cfg = cfg
        self._entities: dict[str, Entity] = {}

        # Build the MjSpec
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.002

        if cfg.light:
            spec.worldbody.add_light(name="main_light", pos=[0, 0, 4], dir=[0, 0, -1])

        if cfg.ground_plane:
            spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.01],
                rgba=[0.8, 0.8, 0.8, 1.0],
            )

        # Attach each entity MJCF
        for name, entity_cfg in cfg.entities.items():
            entity = Entity(entity_cfg, name)
            self._entities[name] = entity

            if entity_cfg.spawn is not None:
                spawn_path = entity_cfg.spawn
                if not os.path.isabs(spawn_path):
                    # Resolve relative to cwd
                    spawn_path = os.path.join(os.getcwd(), spawn_path)
                child_spec = mujoco.MjSpec.from_file(spawn_path)
                prefix = f"{name}/"
                frame = spec.worldbody.add_frame()
                spec.attach(child_spec, prefix=prefix, frame=frame)

        self._spec = spec
        self._model: mujoco.MjModel | None = None

    # ------------------------------------------------------------------
    # Model compilation
    # ------------------------------------------------------------------

    def compile(self) -> mujoco.MjModel:
        """Compile the ``MjSpec`` into a ``MjModel`` and resolve entity indexing."""
        self._model = self._spec.compile()

        # Resolve entity index arrays from the compiled model
        for name, entity in self._entities.items():
            prefix = f"{name}/"
            entity.resolve(self._model, prefix)

        return self._model

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            return self.compile()
        return self._model

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec

    # ------------------------------------------------------------------
    # Entity access
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Entity:
        """Get an entity by name."""
        try:
            return self._entities[name]
        except KeyError as e:
            raise KeyError(
                f"Entity '{name}' not found in scene. Available: {list(self._entities)}"
            ) from e

    def __contains__(self, name: str) -> bool:
        return name in self._entities

    @property
    def entities(self) -> dict[str, Entity]:
        return self._entities

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        entity_names = list(self._entities.keys())
        return f"Scene(num_envs={self.cfg.num_envs}, entities={entity_names})"
