from .v1_augobs import build_model_v1
from .v2_stackedobs import build_model_v2
from .v3_context import (
    build_model_v3,
    build_model_v31,
    build_model_v32,
    build_model_v4,
    build_model_v41,
    build_model_v42_phase,
    build_model_v42_mean,
    build_model_v43_ent,
    build_model_v43_logit_bias,
    build_model_v51_abppo,
    build_model_v52_qcritic,
    build_model_v53_auxweak,
    build_model_v61_cvarppo,
    build_model_v62_v3cvar,
    build_model_v63_cadm,
    build_model_v71_poolppo,
    build_model_v72_poolv3,
    build_model_v73_tcrppo,
    build_model_v74_tcrv3,
    build_model_v8_a,
    build_model_v9_a,
    build_model_v8_a2,
    build_model_v9_a2,
    build_model_v10_a,
    build_model_v10_b,
    build_model_v10_c,
    build_model_v10_d,
    build_model_v10_e,
)


def build_model(env, seed, device, algo_version="v1", **kwargs):
    version = str(algo_version or "v1").strip().lower()
    if version == "v1":
        return build_model_v1(env=env, seed=seed, device=device, **kwargs)
    if version == "v2":
        return build_model_v2(env=env, seed=seed, device=device, **kwargs)
    if version == "v3":
        return build_model_v3(env=env, seed=seed, device=device, **kwargs)
    if version in ("v3.1", "v31", "v3_1"):
        return build_model_v31(env=env, seed=seed, device=device, **kwargs)
    if version in ("v3.2", "v32", "v3_2"):
        return build_model_v32(env=env, seed=seed, device=device, **kwargs)
    if version == "v4":
        return build_model_v4(env=env, seed=seed, device=device, **kwargs)
    if version in ("v4.1", "v41", "v4_1"):
        return build_model_v41(env=env, seed=seed, device=device, **kwargs)
    if version in ("v4.2_phase", "v42_phase", "v4_2_phase"):
        return build_model_v42_phase(env=env, seed=seed, device=device, **kwargs)
    if version in ("v4.2_mean", "v42_mean", "v4_2_mean"):
        return build_model_v42_mean(env=env, seed=seed, device=device, **kwargs)
    if version in ("v4.3_ent", "v43_ent", "v4_3_ent"):
        return build_model_v43_ent(env=env, seed=seed, device=device, **kwargs)
    if version in ("v4.3_logit_bias", "v43_logit_bias", "v4_3_logit_bias"):
        return build_model_v43_logit_bias(env=env, seed=seed, device=device, **kwargs)
    if version in ("v5.1_abppo", "v51_abppo", "v5_1_abppo"):
        return build_model_v51_abppo(env=env, seed=seed, device=device, **kwargs)
    if version in ("v5.2_qcritic", "v52_qcritic", "v5_2_qcritic"):
        return build_model_v52_qcritic(env=env, seed=seed, device=device, **kwargs)
    if version in ("v5.3_auxweak", "v53_auxweak", "v5_3_auxweak"):
        return build_model_v53_auxweak(env=env, seed=seed, device=device, **kwargs)
    if version in ("v6.1_cvarppo", "v61_cvarppo", "v6_1_cvarppo"):
        return build_model_v61_cvarppo(env=env, seed=seed, device=device, **kwargs)
    if version in ("v6.2_v3cvar", "v62_v3cvar", "v6_2_v3cvar"):
        return build_model_v62_v3cvar(env=env, seed=seed, device=device, **kwargs)
    if version in ("v6.3_cadm", "v63_cadm", "v6_3_cadm"):
        return build_model_v63_cadm(env=env, seed=seed, device=device, **kwargs)
    if version in ("v7.1_poolppo", "v71_poolppo", "v7_1_poolppo"):
        return build_model_v71_poolppo(env=env, seed=seed, device=device, **kwargs)
    if version in ("v7.2_poolv3", "v72_poolv3", "v7_2_poolv3"):
        return build_model_v72_poolv3(env=env, seed=seed, device=device, **kwargs)
    if version in ("v7.3_tcrppo", "v73_tcrppo", "v7_3_tcrppo"):
        return build_model_v73_tcrppo(env=env, seed=seed, device=device, **kwargs)
    if version in ("v7.4_tcrv3", "v74_tcrv3", "v7_4_tcrv3"):
        return build_model_v74_tcrv3(env=env, seed=seed, device=device, **kwargs)
    if version in ("v8_a", "v8-a", "v8a", "pponew_v8_a", "pponewv8_a"):
        return build_model_v8_a(env=env, seed=seed, device=device, **kwargs)
    if version in ("v9_a", "v9-a", "v9a", "pponew_v9_a", "pponewv9_a"):
        return build_model_v9_a(env=env, seed=seed, device=device, **kwargs)
    if version in ("v8_a2", "v8-a2", "v8a2", "pponew_v8_a2", "pponewv8_a2"):
        return build_model_v8_a2(env=env, seed=seed, device=device, **kwargs)
    if version in ("v9_a2", "v9-a2", "v9a2", "pponew_v9_a2", "pponewv9_a2"):
        return build_model_v9_a2(env=env, seed=seed, device=device, **kwargs)
    if version in ("v10_a", "v10-a", "v10a", "pponew_v10_a", "pponewv10_a"):
        return build_model_v10_a(env=env, seed=seed, device=device, **kwargs)
    if version in ("v10_b", "v10-b", "v10b", "pponew_v10_b", "pponewv10_b"):
        return build_model_v10_b(env=env, seed=seed, device=device, **kwargs)
    if version in ("v10_c", "v10-c", "v10c", "pponew_v10_c", "pponewv10_c"):
        return build_model_v10_c(env=env, seed=seed, device=device, **kwargs)
    if version in ("v10_d", "v10-d", "v10d", "pponew_v10_d", "pponewv10_d"):
        return build_model_v10_d(env=env, seed=seed, device=device, **kwargs)
    if version in ("v10_e", "v10-e", "v10e", "pponew_v10_e", "pponewv10_e"):
        return build_model_v10_e(env=env, seed=seed, device=device, **kwargs)
    raise ValueError(f"Unsupported PPO_NEW algo_version: {algo_version}")
