# KD4RL Clean Package

This folder is a runnable, reduced extraction of the KD4RL research code.

## Main Entrypoints

- `main_SH.py`: A-share / SH chain.
- `main_Nas.py`: Nasdaq chain.
- Both entrypoints explicitly apply `utils/config_SH.py` or `utils/config_Nas.py` before running.
- `SSM_pipeline.py`: SSM training/inference pipeline used by the commented main paths.
- `Train/PPO_train.py`: hierarchical PPO training/testing entry used by both main chains.
- `Train/baseline.py`: traditional baseline runner.
- `DeepTrader/src/run.py` and `DeepAries/main.py`: deep baselines.

## Included Data

- `Dataset/*/feature`: feature CSVs for baselines and data builders.
- `Dataset/*/feature_ssm`: SSM-enhanced CSVs plus `*_ssm3_states.pt` for PPO.
- `DeepTrader/src/data/NAS`, `DeepTrader/src/data/SH`, `DeepAries/data/nas`, `DeepAries/data/sh`: prepared deep baseline inputs.

## Excluded

Historical checkpoints, generated plots, experiment outputs, raw source datasets, old SAC/DDPG/Hier training branches, old SSM variants, JumpModel examples, and unused config variants are intentionally excluded.

## Smoke Test

Run from this directory:

```bat
call D:\anaconda\Scripts\activate.bat xuangu
python smoke_test.py
```

The smoke test loads both Nasdaq and A-share configs, creates `PPO_Env`, resets it, and executes one equal-weight step for each market.
