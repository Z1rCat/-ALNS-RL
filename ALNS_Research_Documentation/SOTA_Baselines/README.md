# SOTA Baselines (Downloaded)

This folder stores papers and reference repositories for baseline comparison with `NOVA_EDRL`.

## 1) RARL (Robust Adversarial Reinforcement Learning)
- Paper: ICML 2017, Pinto et al.
- PDF source: https://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf
- Local PDF: `RARL/papers/ICML2017_Robust_Adversarial_Reinforcement_Learning.pdf`
- Repo source (community implementation): https://github.com/Jekyll1021/RARL
- Local repo: `RARL/repos/RARL_repo`
- System integration label: `RARL` (master -> outer reference pipeline, objective mode `rarl`)

## 2) PLR / UED (Prioritized Level Replay)
- Paper: ICML 2021, Jiang et al.
- PDF source: https://proceedings.mlr.press/v139/jiang21b/jiang21b.pdf
- Local PDF: `PLR_UED/papers/ICML2021_Prioritized_Level_Replay.pdf`
- Repo source (official): https://github.com/facebookresearch/level-replay
- Local repo: `PLR_UED/repos/level-replay`
- System integration label: `PLR_UED` (aliases: `PLR`, `UED`; objective mode `plr` + curriculum enabled)

## 3) CQL (Conservative Q-Learning)
- Paper: NeurIPS 2020, Kumar et al.
- PDF source: https://papers.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf
- Local PDF: `CQL/papers/NeurIPS2020_Conservative_Q_Learning.pdf`
- Repo source (official): https://github.com/aviralkumar2907/CQL
- Local repo: `CQL/repos/CQL`
- System integration label: `CQL_DQN` (alias: `CQL`; inner RL uses discrete CQL agent)

## 4) CaDM (Context-aware Dynamics Model)
- Paper: ICML 2020, Lee et al.
- PDF source: https://proceedings.mlr.press/v119/lee20g/lee20g.pdf
- Local PDF: `CADM/papers/ICML2020_Context-aware_Dynamics_Model.pdf`
- Repo source: https://github.com/younggyoseo/CaDM
- Local repo: `CADM/repos/CaDM`
- System integration label: `CADM` (master maps to `PPO_LSTM` + context-profile env)

## Notes
- Repository snapshots are cloned with `--depth 1`.
- These baselines are integrated as peer options in `codes/Dynamic_master34959.py` algorithm menu.
