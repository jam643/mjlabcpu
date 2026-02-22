"""Shared pytest fixtures."""

from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

# Path to the cartpole asset
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "assets")
CARTPOLE_XML = os.path.join(ASSET_DIR, "cartpole.xml")


@pytest.fixture
def cartpole_model() -> mujoco.MjModel:
    """Load the cartpole MJCF directly (no prefix) for testing sim/state."""
    return mujoco.MjModel.from_xml_path(CARTPOLE_XML)


@pytest.fixture
def cartpole_data(cartpole_model) -> mujoco.MjData:
    return mujoco.MjData(cartpole_model)


@pytest.fixture
def num_envs() -> int:
    return 3
