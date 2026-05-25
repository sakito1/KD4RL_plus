import multiprocessing
import random

import numpy as np

import utils.config as runtime_config
import utils.config_SH as market_config
from SSM_pipeline import full_pipeline
from Train.PPO_train import main as PPO_train
from Train.baseline import baseline
from utils.Log import create_logger


def apply_market_config():
    """Load the A-share config into the shared utils.config module."""
    for name, value in vars(market_config).items():
        if not name.startswith("__"):
            setattr(runtime_config, name, value)


def run_ppo_batch(cun_path, seed_list):
    PPO_train(cun_path, None, seed_list=seed_list)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    apply_market_config()
    cun_path = "checkpoints/sh"
    logger = create_logger(cun_path)

    # Optional stages:
    # full_pipeline(cun_path, logger, K=2, horizon=1, do_train=True)
    # baseline(cun_path, logger)

    start_seed = 77
    num_branches = 4
    seeds_per_branch = 4

    processes = []
    print(
        f"Start parallel PPO training: {num_branches} branches, seeds "
        f"{start_seed} to {start_seed + num_branches * seeds_per_branch - 1}"
    )

    for i in range(num_branches):
        b_start = start_seed + i * seeds_per_branch
        b_end = b_start + seeds_per_branch
        seed_batch = list(range(b_start, b_end))

        p = multiprocessing.Process(target=run_ppo_batch, args=(cun_path, seed_batch))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All parallel PPO branches finished.")
