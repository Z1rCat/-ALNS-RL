import numpy as np

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def build_model_v1(env, seed, device="cpu", **kwargs):
    if PPO is None:
        raise ImportError("stable_baselines3 is required for PPO_NEW v1.")
    ppo_kwargs = dict(kwargs or {})
    policy = ppo_kwargs.pop("policy", "MlpPolicy")
    ppo_kwargs.setdefault("n_steps", 10)
    ppo_kwargs.setdefault("verbose", 1)
    ppo_kwargs.setdefault("policy_kwargs", None)
    return PPO(policy, env, device=device, seed=seed, **ppo_kwargs)


def compose_augmented_obs(o_t, prev_o, stage_bit, prev_action, prev_reward):
    obs_now = np.asarray(o_t, dtype=np.float32).reshape(-1)
    obs_prev = np.asarray(prev_o, dtype=np.float32).reshape(-1)
    delta_o = obs_now - obs_prev
    return np.concatenate(
        [
            obs_now,
            np.array([float(stage_bit)], dtype=np.float32),
            np.array([float(prev_action)], dtype=np.float32),
            np.array([float(prev_reward)], dtype=np.float32),
            delta_o.astype(np.float32),
        ]
    ).astype(np.float32)


def run_v1_self_check():
    # reset: prev_action=0, prev_reward=0, delta=0
    o0 = np.array([10.0, 2.0], dtype=np.float32)
    x0 = compose_augmented_obs(o_t=o0, prev_o=o0, stage_bit=0.0, prev_action=0.0, prev_reward=0.0)
    assert x0.shape[0] == 7
    assert float(x0[3]) == 0.0
    assert float(x0[4]) == 0.0
    assert np.allclose(x0[-2:], np.array([0.0, 0.0], dtype=np.float32))

    # step-1: x1 uses previous cache (a0=0,r0=0), delta=o1-o0
    o1 = np.array([11.5, 1.0], dtype=np.float32)
    x1 = compose_augmented_obs(o_t=o1, prev_o=o0, stage_bit=1.0, prev_action=0.0, prev_reward=0.0)
    assert float(x1[3]) == 0.0
    assert float(x1[4]) == 0.0
    assert np.allclose(x1[-2:], o1 - o0)

    # step-2: after cache update by step-1 -> prev_action=1, prev_reward=0.7
    o2 = np.array([9.5, 3.0], dtype=np.float32)
    x2 = compose_augmented_obs(o_t=o2, prev_o=o1, stage_bit=0.0, prev_action=1.0, prev_reward=0.7)
    assert float(x2[3]) == 1.0
    assert np.isclose(float(x2[4]), 0.7)
    assert np.allclose(x2[-2:], o2 - o1)

    return True
