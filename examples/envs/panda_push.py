"""Franka Panda planar pushing task — env config for view.py / train.py.

Prerequisites:
    uv run python examples/download_panda_assets.py
"""

from __future__ import annotations

import os
import pathlib
import sys

ASSET_DIR = pathlib.Path(__file__).parent.parent / "assets" / "panda_push"
PANDA_XML = ASSET_DIR / "panda_nohand.xml"
PUCK_XML = ASSET_DIR / "puck.xml"

if not PANDA_XML.exists():
    print(
        "Panda assets not found. Run:\n"
        "    uv run python examples/download_panda_assets.py",
        file=sys.stderr,
    )
    sys.exit(1)

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
from mjlabcpu.training import PPOCfg

DT = 0.002
DECIMATION = 4
EPISODE_LENGTH_S = 8.0
MAX_EPISODE_STEPS = int(EPISODE_LENGTH_S / (DT * DECIMATION))  # 1000

PANDA_CFG = SceneEntityCfg(name="panda")
EEF_CFG = SceneEntityCfg(name="panda", body_names=["link7"])
PUCK_CFG = SceneEntityCfg(name="puck")


def make_env(num_envs: int = 1, render_mode: str | None = None) -> ManagerBasedRlEnv:
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            ground_plane=True,
            entities={
                "panda": EntityCfg(prim_path="panda", spawn=str(PANDA_XML)),
                "puck": EntityCfg(prim_path="puck", spawn=str(PUCK_XML)),
            },
        ),
        sim=SimulationCfg(dt=DT),
        episode_length_s=EPISODE_LENGTH_S,
        decimation=DECIMATION,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(
                        func=obs_mdp.joint_pos_rel,
                        params={"entity_cfg": PANDA_CFG},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_mdp.joint_vel_rel,
                        params={"entity_cfg": PANDA_CFG},
                    ),
                    "eef_pos_xy": ObservationTermCfg(
                        func=obs_mdp.body_pos_w_xy,
                        params={"entity_cfg": EEF_CFG},
                    ),
                    "puck_pos": ObservationTermCfg(
                        func=obs_mdp.root_pos_w,
                        params={"entity_cfg": PUCK_CFG},
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
                params={"object_entity_cfg": PUCK_CFG, "command_name": "goal_pos"},
                weight=5.0,
            ),
            "approach": RewardTermCfg(
                func=rew_mdp.eef_to_object,
                params={"eef_entity_cfg": EEF_CFG, "object_entity_cfg": PUCK_CFG},
                weight=1.0,
            ),
            "action_penalty": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01),
            "vel_penalty": RewardTermCfg(
                func=rew_mdp.joint_vel_l2,
                params={"entity_cfg": PANDA_CFG},
                weight=-0.001,
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": MAX_EPISODE_STEPS},
                time_out=True,
            ),
            "puck_out": TerminationTermCfg(
                func=term_mdp.object_out_of_bounds,
                params={"entity_cfg": PUCK_CFG, "max_xy_dist": 1.5},
                time_out=False,
            ),
        },
        actions={
            "arm": ActionTermCfg(
                cls=JointPositionAction,
                params={"entity_cfg": PANDA_CFG, "scale": 0.3, "use_default_offset": True},
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
    return ManagerBasedRlEnv(cfg, render_mode=render_mode)


def ppo_cfg(num_envs: int) -> PPOCfg:
    return PPOCfg(
        num_steps=2048,
        num_envs=num_envs,
        learning_rate=3e-4,
        num_epochs=10,
        num_minibatches=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        hidden_sizes=(256, 128),
        log_interval=5,
    )
