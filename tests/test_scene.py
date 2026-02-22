"""Tests for Scene and Entity."""

from __future__ import annotations

import os

import pytest

from mjlabcpu.entity import EntityCfg
from mjlabcpu.scene import Scene, SceneCfg

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "assets")
CARTPOLE_XML = os.path.join(ASSET_DIR, "cartpole.xml")


class TestScene:
    def test_compile_no_entities(self):
        cfg = SceneCfg(num_envs=1, ground_plane=True)
        scene = Scene(cfg)
        model = scene.model
        assert model is not None
        assert model.nbody >= 1  # at least world body

    def test_compile_with_entity(self):
        cfg = SceneCfg(
            num_envs=1,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=CARTPOLE_XML)},
        )
        scene = Scene(cfg)
        model = scene.model
        assert model is not None
        assert model.nbody > 1

    def test_entity_access(self):
        cfg = SceneCfg(
            num_envs=2,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=CARTPOLE_XML)},
        )
        scene = Scene(cfg)
        _ = scene.model  # trigger compile

        entity = scene["cartpole"]
        assert entity is not None
        assert entity.name == "cartpole"

    def test_entity_not_found(self):
        cfg = SceneCfg(num_envs=1)
        scene = Scene(cfg)
        with pytest.raises(KeyError):
            _ = scene["nonexistent"]

    def test_entity_indexing_resolved(self):
        cfg = SceneCfg(
            num_envs=1,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=CARTPOLE_XML)},
        )
        scene = Scene(cfg)
        _ = scene.model  # trigger compile + resolve

        entity = scene["cartpole"]
        idx = entity.indexing

        # Cartpole has 2 joints: slider + hinge
        assert len(idx.qpos_addrs) >= 2
        assert len(idx.qvel_addrs) >= 2
        assert len(idx.actuator_ids) >= 1  # one motor

    def test_entity_contains(self):
        cfg = SceneCfg(
            num_envs=1,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=CARTPOLE_XML)},
        )
        scene = Scene(cfg)
        assert "cartpole" in scene
        assert "robot" not in scene
