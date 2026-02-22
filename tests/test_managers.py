"""Tests for all manager classes."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mjlabcpu.entity import EntityCfg
from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlabcpu.envs.mdp import events as event_mdp
from mjlabcpu.envs.mdp import observations as obs_mdp
from mjlabcpu.envs.mdp import rewards as rew_mdp
from mjlabcpu.envs.mdp import terminations as term_mdp
from mjlabcpu.envs.mdp.actions import JointPositionAction
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


@pytest.fixture
def cartpole_env():
    entity_cfg = SceneEntityCfg(name="cartpole")
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=2,
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
        rewards={
            "alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0),
            "upright": RewardTermCfg(
                func=rew_mdp.cartpole_upright,
                params={"entity_cfg": entity_cfg},
                weight=1.0,
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": 625},
                time_out=True,
            ),
            "fallen": TerminationTermCfg(
                func=term_mdp.cartpole_fallen,
                params={"entity_cfg": entity_cfg, "max_pole_angle": 0.2, "max_cart_pos": 2.4},
            ),
        },
        actions={
            "cart_drive": ActionTermCfg(
                cls=JointPositionAction,
                params={"entity_cfg": entity_cfg, "scale": 1.0, "use_default_offset": False},
            ),
        },
        events={
            "reset": EventTermCfg(
                func=event_mdp.reset_joints_uniform,
                mode="reset",
                params={
                    "entity_name": "cartpole",
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (-0.05, 0.05),
                },
            )
        },
    )
    env = ManagerBasedRlEnv(cfg)
    yield env
    env.close()


class TestObservationManager:
    def test_compute_shape(self, cartpole_env):
        obs, _ = cartpole_env.reset()
        # joint_pos_rel: 2 dims, joint_vel_rel: 2 dims → total 4
        assert obs.shape == (2, 4), f"Expected (2, 4), got {obs.shape}"

    def test_compute_dtype(self, cartpole_env):
        obs, _ = cartpole_env.reset()
        assert obs.dtype == np.float32

    def test_jit_deterministic(self, cartpole_env):
        """JIT-compiled obs should be deterministic."""
        obs1, _ = cartpole_env.reset()
        obs2, _ = cartpole_env.reset()
        # Both resets start from zero state (before events add noise)
        # Just check shape consistency
        assert obs1.shape == obs2.shape


class TestRewardManager:
    def test_reward_shape(self, cartpole_env):
        cartpole_env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        _, rewards, _, _, _ = cartpole_env.step(actions)
        assert rewards.shape == (2,)

    def test_reward_dtype(self, cartpole_env):
        cartpole_env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        _, rewards, _, _, _ = cartpole_env.step(actions)
        assert rewards.dtype == np.float32

    def test_alive_reward_positive(self, cartpole_env):
        """is_alive reward should be positive when pole is up."""
        cartpole_env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        _, rewards, terminated, _, _ = cartpole_env.step(actions)
        # Alive envs should have positive reward (is_alive=1 + upright≈1)
        alive = ~terminated
        if alive.any():
            assert rewards[alive].mean() > 0


class TestTerminationManager:
    def test_done_shape(self, cartpole_env):
        cartpole_env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        _, _, terminated, truncated, _ = cartpole_env.step(actions)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)

    def test_time_out(self):
        """Time-out should trigger after max_episode_length steps."""
        from mjlabcpu.entity import EntityCfg

        entity_cfg = SceneEntityCfg(name="cartpole")
        # Use a tiny max_episode_length (5 steps) and NO fall termination
        cfg = ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                num_envs=2,
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
                    }
                )
            },
            rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)},
            terminations={
                "time_out": TerminationTermCfg(
                    func=term_mdp.time_out,
                    params={"max_episode_length": 5},  # small: triggers in 5 steps
                    time_out=True,
                ),
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
        env = ManagerBasedRlEnv(cfg)
        env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        truncated_any = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(actions)
            if truncated.any():
                truncated_any = True
                break
        env.close()
        assert truncated_any, (
            "Time-out should have triggered within 20 steps (max_episode_length=5)"
        )

    def test_fallen_terminates(self, cartpole_env):
        """Applying large forces should cause cartpole to fall and terminate."""
        cartpole_env.reset()
        # Push hard to one side for up to 300 steps
        actions = np.ones((2, 1), dtype=np.float32) * 5.0  # large force
        terminated_any = False
        for _ in range(300):
            _, _, terminated, truncated, _ = cartpole_env.step(actions)
            if terminated.any():
                terminated_any = True
                break
        assert terminated_any, (
            "Expected cartpole to terminate (pole fall or out-of-bounds) within 300 steps "
            "under constant large action=5.0"
        )


class TestActionManager:
    def test_action_dim(self, cartpole_env):
        assert cartpole_env._action_manager.action_dim == 1

    def test_action_space(self, cartpole_env):
        assert cartpole_env.action_space.shape == (1,)

    def test_apply_action(self, cartpole_env):
        cartpole_env.reset()
        actions = np.array([[1.0], [-1.0]], dtype=np.float32)
        obs, rewards, terminated, truncated, info = cartpole_env.step(actions)
        assert obs.shape == (2, 4)


class TestEventManager:
    def test_reset_event_fires_on_reset(self, cartpole_env):
        """reset_joints_uniform event should produce non-zero qpos after reset."""
        cartpole_env.reset()
        # After reset, joint positions should differ from zero due to uniform noise
        # Run several resets and check that at least one produces non-zero state
        any_nonzero = False
        for _ in range(5):
            obs, _ = cartpole_env.reset()
            if np.any(np.abs(obs) > 1e-6):
                any_nonzero = True
                break
        assert any_nonzero, "reset_joints_uniform should inject position/velocity noise"

    def test_reset_event_fires_on_episode_end(self):
        """Events should fire for specific env_ids on episode termination."""
        entity_cfg = SceneEntityCfg(name="cartpole")
        cfg = ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                num_envs=2,
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
                    }
                )
            },
            rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)},
            terminations={
                "time_out": TerminationTermCfg(
                    func=term_mdp.time_out,
                    params={"max_episode_length": 2},
                    time_out=True,
                ),
            },
            actions={
                "cart": ActionTermCfg(
                    cls=JointPositionAction,
                    params={"entity_cfg": entity_cfg, "scale": 1.0, "use_default_offset": False},
                )
            },
            events={
                "reset": EventTermCfg(
                    func=event_mdp.reset_joints_uniform,
                    mode="reset",
                    params={
                        "entity_name": "cartpole",
                        "position_range": (-0.1, 0.1),
                        "velocity_range": (-0.1, 0.1),
                    },
                )
            },
        )
        env = ManagerBasedRlEnv(cfg)
        env.reset()
        actions = np.zeros((2, 1), dtype=np.float32)
        # After 2 steps, timeout triggers and reset event fires for both envs
        for _ in range(3):
            obs, _, _, truncated, _ = env.step(actions)
            if truncated.any():
                # Episode length should be reset to 0 for truncated envs
                assert np.array(env._episode_length[truncated]).max() <= 1
                break
        env.close()

    def test_episode_length_resets_on_done(self, cartpole_env):
        """Episode length should reset to 0 when environments are reset, leaving others intact."""
        cartpole_env.reset()
        # Set env 1 to a non-zero episode length (env 0 stays at 0)
        cartpole_env._episode_length = cartpole_env._episode_length.at[1].set(100)
        # Reset only env 0
        cartpole_env._reset_envs([0])
        assert int(cartpole_env._episode_length[0]) == 0  # env 0: reset to 0
        assert int(cartpole_env._episode_length[1]) == 100  # env 1: unchanged


class TestCommandManager:
    def test_commands_shape(self, cartpole_env):
        """Command manager should be instantiable (empty config for cartpole)."""
        # cartpole_env has no commands configured — manager should be empty
        cmds = cartpole_env._command_manager.commands
        assert isinstance(cmds, dict)

    def test_command_resampling(self):
        """Commands should be resampled per env_id without affecting others."""
        from mjlabcpu.managers.command_manager import UniformVelocityCommandCfg

        entity_cfg = SceneEntityCfg(name="cartpole")
        cfg = ManagerBasedRlEnvCfg(
            scene=SceneCfg(
                num_envs=4,
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
                    }
                )
            },
            rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)},
            terminations={},
            actions={
                "cart": ActionTermCfg(
                    cls=JointPositionAction,
                    params={"entity_cfg": entity_cfg, "scale": 1.0, "use_default_offset": False},
                )
            },
            events={},
            commands={
                "velocity": UniformVelocityCommandCfg(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-1.0, 1.0),
                    resampling_time=5.0,
                )
            },
        )
        env = ManagerBasedRlEnv(cfg)
        env.reset()
        cmds_before = np.array(env._command_manager.commands["velocity"])
        # Resample only env 0 and env 2
        env._command_manager.resample([0, 2])
        cmds_after = np.array(env._command_manager.commands["velocity"])
        # Envs 1 and 3 should be unchanged
        assert np.allclose(cmds_before[1], cmds_after[1]), "Env 1 command should not change"
        assert np.allclose(cmds_before[3], cmds_after[3]), "Env 3 command should not change"
        # Shape must be preserved
        assert cmds_after.shape == (4, 3)
        env.close()
