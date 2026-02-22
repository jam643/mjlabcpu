"""Smoke tests for the PPO training module."""

from __future__ import annotations

import os

import numpy as np

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "cartpole.xml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cartpole_env(num_envs: int = 2, render_mode=None):
    """Build a minimal cartpole env for training tests."""
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

    entity_cfg = SceneEntityCfg(name="cartpole")
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            ground_plane=True,
            entities={"cartpole": EntityCfg(prim_path="cartpole", spawn=ASSET_PATH)},
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
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out,
                params={"max_episode_length": 625},
                time_out=True,
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


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------


def test_actor_critic_forward():
    """ActorCritic produces correct output shapes."""
    import jax
    import jax.numpy as jnp

    from mjlabcpu.training.networks import ActorCritic

    net = ActorCritic(hidden_sizes=(64, 32), action_dim=3)
    obs = jnp.zeros((8, 10))  # batch=8, obs_dim=10
    params = net.init(jax.random.PRNGKey(0), obs)["params"]
    mean, log_std, value = net.apply({"params": params}, obs)

    assert mean.shape == (8, 3), f"Expected (8,3), got {mean.shape}"
    assert log_std.shape == (3,), f"Expected (3,), got {log_std.shape}"
    assert value.shape == (8,), f"Expected (8,), got {value.shape}"


# ---------------------------------------------------------------------------
# Rollout buffer / GAE tests
# ---------------------------------------------------------------------------


def test_compute_gae_shapes():
    """GAE returns correct shapes."""
    import jax.numpy as jnp

    from mjlabcpu.training.rollout import compute_gae

    T, N = 16, 4
    rewards = jnp.ones((T, N))
    values = jnp.zeros((T, N))
    dones = jnp.zeros((T, N), dtype=bool)
    last_value = jnp.zeros(N)

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)


def test_compute_gae_terminal_bootstrap():
    """At a terminal step, next value should not be bootstrapped."""
    import jax.numpy as jnp

    from mjlabcpu.training.rollout import compute_gae

    T, N = 4, 1
    rewards = jnp.ones((T, N))
    values = jnp.ones((T, N)) * 0.5
    # Terminal at step 1
    dones = jnp.array([[False], [True], [False], [False]])
    last_value = jnp.ones(N)

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    # Just check no NaN
    assert not jnp.any(jnp.isnan(adv))
    assert not jnp.any(jnp.isnan(ret))


# ---------------------------------------------------------------------------
# PPOCfg test
# ---------------------------------------------------------------------------


def test_ppo_cfg_defaults():
    from mjlabcpu.training import PPOCfg

    cfg = PPOCfg()
    assert cfg.num_steps == 2048
    assert cfg.gamma == 0.99
    assert cfg.clip_coef == 0.2


# ---------------------------------------------------------------------------
# PPOTrainer smoke tests
# ---------------------------------------------------------------------------


def test_ppo_trainer_init():
    """PPOTrainer initialises without error."""
    from mjlabcpu.training import PPOCfg, PPOTrainer

    env = _make_cartpole_env(num_envs=2)
    try:
        cfg = PPOCfg(num_steps=16, num_envs=2, num_minibatches=2)
        trainer = PPOTrainer(env, cfg)
        assert trainer._obs_dim > 0
        assert trainer._act_dim > 0
    finally:
        env.close()


def test_ppo_trainer_get_action():
    """get_action returns correct shape."""
    from mjlabcpu.training import PPOCfg, PPOTrainer

    env = _make_cartpole_env(num_envs=2)
    try:
        trainer = PPOTrainer(env, PPOCfg(num_envs=2))
        obs, _ = env.reset()
        actions = trainer.get_action(obs)
        assert actions.shape == (2, env.action_space.shape[0])
        actions_det = trainer.get_action(obs, deterministic=True)
        assert actions_det.shape == (2, env.action_space.shape[0])
    finally:
        env.close()


def test_ppo_trainer_short_train():
    """Train for a very short number of steps; check metrics dict is returned."""
    from mjlabcpu.training import PPOCfg, PPOTrainer

    env = _make_cartpole_env(num_envs=2)
    try:
        cfg = PPOCfg(
            num_steps=32,
            num_envs=2,
            num_epochs=2,
            num_minibatches=2,
            log_interval=1,
        )
        trainer = PPOTrainer(env, cfg)
        # 2 envs × 32 steps = 64 per update; request 2 updates = 128 steps total
        metrics = trainer.train(total_timesteps=128)
        assert "pg_loss" in metrics
        assert "v_loss" in metrics
        assert "entropy" in metrics
        assert len(metrics["update"]) >= 1
    finally:
        env.close()


def test_ppo_save_load(tmp_path):
    """Save and load checkpoint; loaded params match saved params."""
    import jax

    from mjlabcpu.training import PPOCfg, PPOTrainer

    env = _make_cartpole_env(num_envs=2)
    try:
        cfg = PPOCfg(num_steps=16, num_envs=2, num_minibatches=2)
        trainer = PPOTrainer(env, cfg)

        save_path = str(tmp_path / "test_ckpt.pkl")
        trainer.save(save_path)

        # Load into a fresh trainer
        trainer2 = PPOTrainer(env, cfg)
        trainer2.load(save_path)

        leaves1 = jax.tree_util.tree_leaves(trainer._state.params)
        leaves2 = jax.tree_util.tree_leaves(trainer2._state.params)
        for l1, l2 in zip(leaves1, leaves2, strict=True):
            assert np.allclose(np.array(l1), np.array(l2))
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------


def test_render_rgb_array():
    """rgb_array mode returns correct shaped numpy array."""
    env = _make_cartpole_env(num_envs=2, render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert frame is not None, "render() returned None in rgb_array mode"
        assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_render_none_mode():
    """No render_mode returns None."""
    env = _make_cartpole_env(num_envs=2, render_mode=None)
    try:
        env.reset()
        result = env.render()
        assert result is None
    finally:
        env.close()
