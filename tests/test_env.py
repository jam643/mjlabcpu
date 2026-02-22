"""End-to-end tests for ManagerBasedRlEnv."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mjlabcpu.entity import EntityCfg
from mjlabcpu.envs.mdp import observations as obs_mdp
from mjlabcpu.envs.mdp import rewards as rew_mdp
from mjlabcpu.envs.mdp import terminations as term_mdp
from mjlabcpu.envs.mdp import events as event_mdp
from mjlabcpu.envs.mdp.actions import JointPositionAction
from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlabcpu.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlabcpu.scene import SceneCfg
from mjlabcpu.sim import SimulationCfg

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "assets")
CARTPOLE_XML = os.path.join(ASSET_DIR, "cartpole.xml")


def make_simple_env(num_envs: int = 2) -> ManagerBasedRlEnv:
    entity_cfg = SceneEntityCfg(name="cartpole")
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=CARTPOLE_XML)},
        ),
        sim=SimulationCfg(dt=0.002),
        episode_length_s=5.0,
        decimation=4,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=obs_mdp.joint_pos_rel,
                        params={"entity_cfg": entity_cfg},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_mdp.joint_vel_rel,
                        params={"entity_cfg": entity_cfg},
                    ),
                }
            )
        },
        rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)},
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": 625},
                time_out=True,
            )
        },
        actions={
            "cart": ActionTermCfg(
                cls=JointPositionAction,
                params={"entity_cfg": entity_cfg, "scale": 1.0, "use_default_offset": False},
            )
        },
        events={
            "reset": EventTermCfg(
                func=event_mdp.reset_scene_to_default,
                mode="reset",
            )
        },
    )
    return ManagerBasedRlEnv(cfg)


class TestManagerBasedRlEnv:
    def test_init(self):
        env = make_simple_env(2)
        assert env.num_envs == 2
        env.close()

    def test_reset_returns_correct_shape(self):
        env = make_simple_env(2)
        obs, info = env.reset()
        assert obs.shape == (2, 4)  # 2 qpos + 2 qvel
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_correct_shapes(self):
        num_envs = 3
        env = make_simple_env(num_envs)
        env.reset()
        actions = np.zeros((num_envs, 1), dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs.shape == (num_envs, 4)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert "reward_terms" in info
        assert "termination_terms" in info
        env.close()

    def test_100_steps_no_crash(self):
        """Smoke test: run 100 steps without error."""
        env = make_simple_env(4)
        env.reset()
        actions = np.random.uniform(-1, 1, (4, 1)).astype(np.float32)
        for _ in range(100):
            obs, rewards, terminated, truncated, info = env.step(actions)
        env.close()

    def test_gymnasium_compatible(self):
        """Verify gymnasium spaces are valid."""
        import gymnasium as gym

        env = make_simple_env(1)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        env.close()

    def test_obs_dtype(self):
        env = make_simple_env(2)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_episode_length_resets(self):
        """Episode length counter should reset to 0 on env reset."""
        env = make_simple_env(2)
        env.reset()
        for _ in range(5):
            env.step(np.zeros((2, 1), dtype=np.float32))
        # Manually reset all envs
        env._reset_envs([0, 1])
        ep_len = np.array(env._episode_length)
        assert np.all(ep_len == 0), f"Episode length should be 0 after reset, got {ep_len}"
        env.close()

    def test_multi_env_independent(self):
        """Different actions should lead to different states across envs."""
        env = make_simple_env(2)
        env.reset()
        # Apply opposite actions
        actions = np.array([[1.0], [-1.0]], dtype=np.float32)
        for _ in range(20):
            obs, _, _, _, _ = env.step(actions)
        # Env 0 and env 1 should have different cart positions
        assert not np.allclose(obs[0], obs[1], atol=1e-4), "Envs should diverge with different actions"
        env.close()
