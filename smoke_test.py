import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def dataset_dirs():
    dataset_root = os.path.join(ROOT, "Dataset")
    dirs = [
        os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    ]
    dirs = [d for d in dirs if os.path.isdir(os.path.join(d, "feature_ssm"))]
    nas = next((d for d in dirs if os.path.basename(d).startswith("Nas100")), None)
    sh = next((d for d in dirs if d != nas), None)
    if not nas or not sh:
        raise RuntimeError("Cannot resolve NAS and SH dataset directories.")
    return nas, sh


def main():
    print(f"[smoke] cwd={ROOT}")
    print(f"[smoke] python={sys.executable}")

    require(os.path.join(ROOT, "main_SH.py"))
    require(os.path.join(ROOT, "main_Nas.py"))

    nas_dir, sh_dir = dataset_dirs()
    require(os.path.join(nas_dir, "feature_ssm"))
    require(os.path.join(sh_dir, "feature_ssm"))
    print("[smoke] nas_dataset=resolved")
    print("[smoke] sh_dataset=resolved")

    import main_Nas
    import main_SH

    def run_one_step(label, apply_market_config):
        apply_market_config()
        from env import PPO_Env
        import utils.config as config

        stock_path = config.dataset["stocks_path"]
        ssm_path = config.dataset["ssm_data_path"]
        require(stock_path)
        require(ssm_path)
        print(f"[smoke] {label} stocks={stock_path}")
        print(f"[smoke] {label} ssm_data=resolved")

        env = PPO_Env()
        env.set_mode("test")
        obs = env.reset()

        w = torch.ones(env.num_stocks, dtype=torch.float32, device=env.device) / env.num_stocks
        next_obs, _reward, done, info = env.step(w, w, outer_action=w, is_switch=True)

        assert "ssm" in obs and "outer_state" in obs and "inner_state" in obs
        assert "portfolio_value" in info
        assert next_obs["outer_state"].shape[1] == env.num_stocks

        print(f"[smoke] {label} num_stocks={env.num_stocks}, dates={len(env.all_dates)}, device={env.device}")
        print(f"[smoke] {label} one_step portfolio_value={info['portfolio_value']:.6f}, done={done}")

    run_one_step("nasdaq", main_Nas.apply_market_config)
    run_one_step("a_share", main_SH.apply_market_config)
    print("[smoke] OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
