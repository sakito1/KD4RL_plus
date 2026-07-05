from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "paper_experiments_outputs" / "paper_experiments_final"
PAPER_FIG = ROOT / "paper" / "figures"


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
    }
)


def compose_pair(left: Path, right: Path, out_base: Path, labels, figsize, *, wspace=0.035, top=0.925):
    left_img = mpimg.imread(left)
    right_img = mpimg.imread(right)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, label in zip(axes, [left_img, right_img], labels):
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.000,
            1.018,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
            color="#111827",
        )
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.010, top=top, wspace=wspace)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    labels = ["(a) Nasdaq-100", "(b) CSI-300"]
    compose_pair(
        EXP / "01_main_experiment" / "main_equity_nas.png",
        EXP / "01_main_experiment" / "main_equity_sh.png",
        PAPER_FIG / "main_equity_curves",
        labels,
        figsize=(16.0, 5.1),
        top=0.925,
    )
    compose_pair(
        EXP / "01_main_experiment" / "main_metrics_nas.png",
        EXP / "01_main_experiment" / "main_metrics_sh.png",
        PAPER_FIG / "main_metric_bars",
        labels,
        figsize=(16.0, 5.6),
        top=0.925,
    )
    compose_pair(
        EXP / "02_ablation" / "ablation_metrics_nas.png",
        EXP / "02_ablation" / "ablation_metrics_sh.png",
        PAPER_FIG / "ablation_metric_bars",
        labels,
        figsize=(16.0, 5.6),
        top=0.925,
    )
    compose_pair(
        EXP / "03_controller_interpretability" / "controller_case_nas_01.png",
        EXP / "03_controller_interpretability" / "controller_case_sh_01.png",
        PAPER_FIG / "explainability" / "controller_switch_cases",
        labels,
        figsize=(16.0, 6.4),
        top=0.925,
    )
    compose_pair(
        EXP / "03_controller_interpretability" / "fixed_window_comparison_nas.png",
        EXP / "03_controller_interpretability" / "fixed_window_comparison_sh.png",
        PAPER_FIG / "explainability" / "fixed_window_comparison",
        labels,
        figsize=(16.0, 4.7),
        top=0.910,
    )
    compose_pair(
        EXP / "04_inner_actor_interpretability" / "inner_actor_base_adjustment_future_return_nas.png",
        EXP / "04_inner_actor_interpretability" / "inner_actor_base_adjustment_future_return_sh.png",
        PAPER_FIG / "explainability" / "inner_actor_base_adjustment",
        labels,
        figsize=(16.0, 6.8),
        top=0.925,
    )


if __name__ == "__main__":
    main()
