"""Tests for MJX backend: correctness and speed vs CPU MuJoCo."""

from __future__ import annotations

import os
import time

import jax.numpy as jnp
import numpy as np

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "cartpole.xml")
NUM_ENVS = 64  # Large enough to show MJX batching advantage
SPEED_STEPS = 500  # Steps timed after JIT warmup


# ---------------------------------------------------------------------------
# Shared env factory
# ---------------------------------------------------------------------------


def _make_env(num_envs: int, backend: str, render_mode=None):
    """Build a cartpole env with either 'cpu' or 'mjx' backend."""
    from mjlabcpu.entity import EntityCfg
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
    cfg_kwargs = dict(
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
                        func=obs_mdp.joint_pos_rel, params={"entity_cfg": entity_cfg}
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_mdp.joint_vel_rel, params={"entity_cfg": entity_cfg}
                    ),
                }
            )
        },
        rewards={
            "alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0),
            "upright": RewardTermCfg(
                func=rew_mdp.cartpole_upright, params={"entity_cfg": entity_cfg}, weight=2.0
            ),
            "action_penalty": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out, params={"max_episode_length": 1250}, time_out=True
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
            "reset_cartpole": EventTermCfg(
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

    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

    cfg = ManagerBasedRlEnvCfg(**cfg_kwargs)

    if backend == "cpu":
        from mjlabcpu.envs import ManagerBasedRlEnv

        return ManagerBasedRlEnv(cfg, render_mode=render_mode)
    elif backend == "mjx":
        from mjlabcpu.envs.mjx_env import MjxManagerBasedRlEnv

        return MjxManagerBasedRlEnv(cfg, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_mjx_env_init():
    """MJX env initialises without error."""
    env = _make_env(num_envs=4, backend="mjx")
    try:
        assert env.num_envs == 4
        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape[0] > 0
    finally:
        env.close()


def test_mjx_env_reset_shape():
    """reset() returns correct obs shape."""
    env = _make_env(num_envs=4, backend="mjx")
    try:
        obs, info = env.reset()
        assert obs.shape == (4, env.observation_space.shape[0])
        assert obs.dtype == np.float32
    finally:
        env.close()


def test_mjx_env_step_shapes():
    """step() returns all outputs with correct shapes."""
    N = 4
    env = _make_env(num_envs=N, backend="mjx")
    try:
        env.reset()
        actions = np.zeros((N, env.action_space.shape[0]), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert obs.shape == (N, env.observation_space.shape[0])
        assert reward.shape == (N,)
        assert terminated.shape == (N,)
        assert truncated.shape == (N,)
    finally:
        env.close()


def test_mjx_env_obs_dtype():
    """Observations are float32."""
    env = _make_env(num_envs=2, backend="mjx")
    try:
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
        obs, *_ = env.step(actions)
        assert obs.dtype == np.float32
    finally:
        env.close()


def test_mjx_env_100_steps():
    """100 steps run without error or NaN."""
    env = _make_env(num_envs=4, backend="mjx")
    try:
        obs, _ = env.reset()
        for _ in range(100):
            actions = np.random.uniform(-1, 1, (4, env.action_space.shape[0])).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert not np.any(np.isnan(obs)), "NaN in obs"
            assert not np.any(np.isnan(reward)), "NaN in reward"
    finally:
        env.close()


def test_mjx_env_reset_resets_episode_length():
    """Episode length counter resets when done."""
    env = _make_env(num_envs=4, backend="mjx")
    try:
        env.reset()
        for _ in range(10):
            actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
            env.step(actions)
        # Episode length should have increased
        assert jnp.any(env._episode_length > 0)
        env.reset()
        assert jnp.all(env._episode_length == 0)
    finally:
        env.close()


def test_mjx_render_rgb_array():
    """rgb_array render returns correct shape."""
    env = _make_env(num_envs=2, backend="mjx", render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_mjx_obs_matches_cpu():
    """MJX and CPU envs produce similar obs from the same starting state.

    Both envs start from default state (no randomisation) and take zero
    actions. Physics should be equivalent, so obs should be close.
    """
    N = 1

    from mjlabcpu.entity import EntityCfg
    from mjlabcpu.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
    from mjlabcpu.envs.mdp import observations as obs_mdp
    from mjlabcpu.envs.mdp import rewards as rew_mdp
    from mjlabcpu.envs.mdp import terminations as term_mdp
    from mjlabcpu.envs.mdp.actions import JointPositionAction
    from mjlabcpu.managers import (
        ActionTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from mjlabcpu.scene import SceneCfg
    from mjlabcpu.sim import SimulationCfg

    entity_cfg = SceneEntityCfg(name="cartpole")
    # No events (no randomisation) so both start from exact same state
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=N,
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
                        func=obs_mdp.joint_pos_rel, params={"entity_cfg": entity_cfg}
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_mdp.joint_vel_rel, params={"entity_cfg": entity_cfg}
                    ),
                }
            )
        },
        rewards={"alive": RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)},
        terminations={
            "time_out": TerminationTermCfg(
                func=term_mdp.time_out, params={"max_episode_length": 1250}, time_out=True
            )
        },
        actions={
            "cart_drive": ActionTermCfg(
                cls=JointPositionAction,
                params={"entity_cfg": entity_cfg, "scale": 1.0, "use_default_offset": False},
            )
        },
        events={},  # no randomisation
    )

    from mjlabcpu.envs import ManagerBasedRlEnv
    from mjlabcpu.envs.mjx_env import MjxManagerBasedRlEnv

    cpu_env = ManagerBasedRlEnv(cfg)
    mjx_env = MjxManagerBasedRlEnv(cfg)
    try:
        cpu_obs, _ = cpu_env.reset()
        mjx_obs, _ = mjx_env.reset()

        # Both start from zero state — obs should be identical
        np.testing.assert_allclose(cpu_obs, mjx_obs, atol=1e-5)

        # Take 20 zero-action steps and compare
        actions = np.zeros((N, cpu_env.action_space.shape[0]), dtype=np.float32)
        for _ in range(20):
            cpu_obs, *_ = cpu_env.step(actions)
            mjx_obs, *_ = mjx_env.step(actions)

        np.testing.assert_allclose(cpu_obs, mjx_obs, atol=1e-4)
    finally:
        cpu_env.close()
        mjx_env.close()


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------


def _measure_sps(env, n_steps: int) -> float:
    """Return env-steps/sec after JIT warmup (first step excluded)."""
    obs, _ = env.reset()
    act_dim = env.action_space.shape[0]
    N = env.num_envs

    # Warmup step (JIT compile)
    actions = np.random.uniform(-1, 1, (N, act_dim)).astype(np.float32)
    env.step(actions)

    # Timed steps
    t0 = time.perf_counter()
    for _ in range(n_steps):
        actions = np.random.uniform(-1, 1, (N, act_dim)).astype(np.float32)
        env.step(actions)
    elapsed = time.perf_counter() - t0

    return N * n_steps / elapsed


def test_mjx_faster_than_cpu():
    """MJX env-steps/sec exceeds CPU for large num_envs.

    With 64 envs, the fused JIT kernel (physics + obs + reward + done + reset
    in one XLA call) should outperform the CPU thread-pool backend.
    """
    cpu_env = _make_env(num_envs=NUM_ENVS, backend="cpu")
    mjx_env = _make_env(num_envs=NUM_ENVS, backend="mjx")
    try:
        cpu_sps = _measure_sps(cpu_env, SPEED_STEPS)
        mjx_sps = _measure_sps(mjx_env, SPEED_STEPS)

        print(f"\nSpeed comparison ({NUM_ENVS} envs, {SPEED_STEPS} steps):")
        print(f"  CPU MuJoCo : {cpu_sps:>10,.0f} env-steps/sec")
        print(f"  MJX        : {mjx_sps:>10,.0f} env-steps/sec")
        print(f"  Speedup    : {mjx_sps / cpu_sps:.2f}x")

        assert mjx_sps > cpu_sps, (
            f"MJX ({mjx_sps:.0f} sps) should be faster than CPU ({cpu_sps:.0f} sps) "
            f"with {NUM_ENVS} envs"
        )
    finally:
        cpu_env.close()
        mjx_env.close()


def test_mjx_absolute_throughput():
    """MJX achieves at least 10k env-steps/sec with 64 envs on CPU-only JAX.

    50k+ is achievable on GPU hardware. On CPU-only JAX (e.g. Apple M-series)
    ~20k sps is typical. The minimum of 10k guards against severe regressions.
    """
    env = _make_env(num_envs=NUM_ENVS, backend="mjx")
    try:
        sps = _measure_sps(env, SPEED_STEPS)
        print(f"\nMJX throughput ({NUM_ENVS} envs): {sps:,.0f} env-steps/sec")
        assert sps > 10_000, f"Expected >10k sps, got {sps:.0f}"
    finally:
        env.close()
