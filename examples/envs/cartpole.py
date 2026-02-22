"""Cartpole balance task — env config for view.py / train.py."""

from __future__ import annotations

import os

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
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlabcpu.scene import SceneCfg
from mjlabcpu.sim import SimulationCfg
from mjlabcpu.training import PPOCfg

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "cartpole.xml")


def make_env(num_envs: int = 1, render_mode: str | None = None) -> ManagerBasedRlEnv:
    entity_cfg = SceneEntityCfg(name="cartpole")
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            ground_plane=True,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=ASSET_PATH)},
        ),
        sim=SimulationCfg(dt=0.002),
        episode_length_s=10.0,
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
                weight=2.0,
            ),
            "action_penalty": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": 1250},
                time_out=True,
            ),
            "fallen": TerminationTermCfg(
                func=term_mdp.cartpole_fallen,
                params={"entity_cfg": entity_cfg, "max_pole_angle": 0.2, "max_cart_pos": 2.4},
                time_out=False,
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
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (-0.1, 0.1),
                },
            ),
        },
    )
    return ManagerBasedRlEnv(cfg, render_mode=render_mode)


def ppo_cfg(num_envs: int) -> PPOCfg:
    return PPOCfg(
        num_steps=512,
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
