import numpy as np

from .v1_augobs import build_model_v1, compose_augmented_obs


def build_model_v2(env, seed, device="cpu", **kwargs):
    # v2 only changes observation construction in env; PPO hyper-params remain aligned with v1/base PPO.
    return build_model_v1(env=env, seed=seed, device=device, **kwargs)


def compose_stacked_obs(x_now, history=None, k=4):
    x_vec = np.asarray(x_now, dtype=np.float32).reshape(-1)
    k = max(1, int(k))
    if k == 1:
        return x_vec
    hist = [] if history is None else [np.asarray(v, dtype=np.float32).reshape(-1) for v in history]
    if not hist:
        hist = [x_vec.copy() for _ in range(k)]
    else:
        hist = [x_vec.copy()] + hist[: k - 1]
        if len(hist) < k:
            hist.extend([x_vec.copy()] * (k - len(hist)))
    return np.concatenate(hist, axis=0).astype(np.float32)


def run_v2_self_check():
    o0 = np.array([10.0, 2.0], dtype=np.float32)
    o1 = np.array([11.5, 1.0], dtype=np.float32)
    o2 = np.array([9.5, 3.0], dtype=np.float32)

    x0 = compose_augmented_obs(o_t=o0, prev_o=o0, stage_bit=0.0, prev_action=0.0, prev_reward=0.0)
    X0 = compose_stacked_obs(x0, history=None, k=4)
    assert X0.shape[0] == x0.shape[0] * 4
    assert np.allclose(X0[: x0.shape[0]], x0)

    x1 = compose_augmented_obs(o_t=o1, prev_o=o0, stage_bit=1.0, prev_action=0.0, prev_reward=0.0)
    X1 = compose_stacked_obs(x1, history=[x0, x0, x0], k=4)
    assert np.allclose(X1[: x1.shape[0]], x1)
    assert np.allclose(X1[x1.shape[0] : 2 * x1.shape[0]], x0)

    x2 = compose_augmented_obs(o_t=o2, prev_o=o1, stage_bit=0.0, prev_action=1.0, prev_reward=0.7)
    X2 = compose_stacked_obs(x2, history=[x1, x0, x0], k=4)
    assert np.isclose(float(x2[3]), 1.0)
    assert np.isclose(float(x2[4]), 0.7)
    assert np.allclose(X2[: x2.shape[0]], x2)
    assert np.allclose(X2[x2.shape[0] : 2 * x2.shape[0]], x1)
    return True
