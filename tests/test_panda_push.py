"""Tests for the Franka Panda planar pushing environment.

Auto-skipped if panda assets have not been downloaded.
Run download first:
    uv run python examples/download_panda_assets.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

_ASSET_DIR = pathlib.Path(__file__).parent.parent / "examples" / "assets" / "panda_push"
PANDA_XML = _ASSET_DIR / "panda_nohand.xml"
PUCK_XML = _ASSET_DIR / "puck.xml"

pytestmark = pytest.mark.skipif(
    not PANDA_XML.exists(),
    reason="Panda assets not found. Run: uv run python examples/download_panda_assets.py",
)


def _make_env(num_envs: int = 2):
    """Build the panda push env inside the function to avoid import errors when skipped."""
    from mjlabcpu.entity import EntityCfg
    from mjlabcpu.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlabcpu.envs.mdp import events as event_mdp
    from mjlabcpu.envs.mdp import observations as obs_mdp
    from mjlabcpu.envs.mdp import rewards as rew_mdp
    from mjlabcpu.envs.mdp import terminations as term_mdp
    from mjlabcpu.envs.mdp.actions import JointPositionAction
    from mjlabcpu.managers import (
        ActionTermCfg,
        EventTermCfg,
        GoalPositionCommandCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from mjlabcpu.scene import SceneCfg
    from mjlabcpu.sim import SimulationCfg

    panda_cfg = SceneEntityCfg(name="panda")
    eef_cfg = SceneEntityCfg(name="panda", body_names=["link7"])
    puck_cfg = SceneEntityCfg(name="puck")

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            ground_plane=True,
            entities={
                "panda": EntityCfg(prim_path="panda", spawn=str(PANDA_XML)),
                "puck": EntityCfg(prim_path="puck", spawn=str(PUCK_XML)),
            },
        ),
        sim=SimulationCfg(dt=0.002),
        episode_length_s=8.0,
        decimation=4,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=obs_mdp.joint_pos_rel,
                        params={"entity_cfg": panda_cfg},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_mdp.joint_vel_rel,
                        params={"entity_cfg": panda_cfg},
                    ),
                    "eef_pos_xy": ObservationTermCfg(
                        func=obs_mdp.body_pos_w_xy,
                        params={"entity_cfg": eef_cfg},
                    ),
                    "puck_pos": ObservationTermCfg(
                        func=obs_mdp.root_pos_w,
                        params={"entity_cfg": puck_cfg},
                    ),
                    "goal_pos": ObservationTermCfg(
                        func=obs_mdp.generated_commands,
                        params={"command_name": "goal_pos"},
                    ),
                }
            )
        },
        rewards={
            "push_progress": RewardTermCfg(
                func=rew_mdp.object_to_goal,
                params={"object_entity_cfg": puck_cfg, "command_name": "goal_pos"},
                weight=5.0,
            ),
            "approach": RewardTermCfg(
                func=rew_mdp.eef_to_object,
                params={"eef_entity_cfg": eef_cfg, "object_entity_cfg": puck_cfg},
                weight=1.0,
            ),
            "action_penalty": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": 1000},
                time_out=True,
            ),
            "puck_out": TerminationTermCfg(
                func=term_mdp.object_out_of_bounds,
                params={"entity_cfg": puck_cfg, "max_xy_dist": 1.5},
                time_out=False,
            ),
        },
        actions={
            "arm": ActionTermCfg(
                cls=JointPositionAction,
                params={"entity_cfg": panda_cfg, "scale": 0.3, "use_default_offset": True},
            ),
        },
        events={
            "reset_panda": EventTermCfg(
                func=event_mdp.reset_joints_uniform,
                mode="reset",
                params={
                    "entity_name": "panda",
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (-0.05, 0.05),
                },
            ),
            "reset_puck": EventTermCfg(
                func=event_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "entity_name": "puck",
                    "pose_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (0.02, 0.02)},
                    "velocity_range": {},
                },
            ),
        },
        commands={
            "goal_pos": GoalPositionCommandCfg(
                x_range=(-0.4, 0.4),
                y_range=(-0.4, 0.4),
                z=0.02,
            ),
        },
    )

    return ManagerBasedRlEnv(cfg)


class TestPandaPushEnv:
    def test_env_init(self):
        env = _make_env(num_envs=2)
        assert env.num_envs == 2
        env.close()

    def test_reset_shape(self):
        env = _make_env(num_envs=2)
        obs, info = env.reset()
        # 7 joint_pos + 7 joint_vel + 2 eef_xy + 3 puck_pos + 3 goal_pos = 22
        assert obs.shape == (2, 22), f"Expected (2, 22), got {obs.shape}"
        assert isinstance(info, dict)
        env.close()

    def test_step_shapes(self):
        num_envs = 3
        env = _make_env(num_envs=num_envs)
        env.reset()
        # Panda has 7 DOF (nohand)
        actions = np.zeros((num_envs, 7), dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs.shape == (num_envs, 22)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        env.close()

    def test_50_steps_no_nan(self):
        env = _make_env(num_envs=2)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), "NaN in initial observation"
        for step in range(50):
            actions = np.random.uniform(-0.1, 0.1, (2, 7)).astype(np.float32)
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {step}"
            assert not np.any(np.isnan(rewards)), f"NaN in rewards at step {step}"
        env.close()

    def test_goal_position_command_sampling(self):
        """GoalPositionCommandCfg should resample valid XY goals on reset."""
        from mjlabcpu.managers import GoalPositionCommandCfg

        cfg = GoalPositionCommandCfg(x_range=(-0.4, 0.4), y_range=(-0.4, 0.4), z=0.02)
        assert cfg.x_range == (-0.4, 0.4)
        assert cfg.y_range == (-0.4, 0.4)
        assert cfg.z == 0.02

        env = _make_env(num_envs=4)
        env.reset()
        # Commands should be populated after reset
        goal = np.array(env._command_manager.commands["goal_pos"])
        assert goal.shape == (4, 3), f"Expected (4, 3), got {goal.shape}"
        # X and Y within range
        assert np.all(goal[:, 0] >= -0.4) and np.all(goal[:, 0] <= 0.4)
        assert np.all(goal[:, 1] >= -0.4) and np.all(goal[:, 1] <= 0.4)
        # Z fixed
        assert np.allclose(goal[:, 2], 0.02)
        env.close()
