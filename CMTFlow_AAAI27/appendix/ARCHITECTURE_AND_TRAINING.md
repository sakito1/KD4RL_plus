# Appendix A — Architecture and Training Details

This file maps the Appendix A subsections to the minimum implementation supplied
in the parent reproduction package. The two markets use the same architecture
and five-stage training procedure; only market data, universe size, seed, and
market-specific configuration differ.

## A.1 Daily decision workflow

The daily environment transition, drifted holdings, proportional transaction
cost, and portfolio reward are implemented in:

- `../src/env/PPO_env.py`
- `../src/agent/PPO_agent.py`
- `../src/run_hrl_training.py`

The Manager proposes a structural portfolio, the Controller decides whether to
retain the current portfolio or adopt the candidate, and the Trader applies a
daily within-support refinement.

## A.2 Manager and Trader architectures

Network definitions and forward passes are in:

- `../src/Components/PPO_model.py`
- `../src/Train/PPO_train.py`

The Manager operates on the longer 60-day lookback and produces a top-K
structural allocation. The Trader uses the shorter 10-day lookback and refines
weights only within the Manager-selected support.

## A.3 Controller architecture

The Controller encoder, Base reconstruction-need head, and switching-advantage
head are defined in `../src/Components/PPO_model.py`. Controller optimization
and trajectory collection are implemented by:

- `../src/Train/controller_pg.py`
- `../src/Train/PPO_train.py`

The final switching logit combines the Base term with a bounded
switching-advantage correction.

## A.4 Training objectives

The role-aligned reward and loss implementation is in
`../src/Train/PPO_train.py`:

- Manager: segment-level structural portfolio outcome.
- Trader: daily incremental cost-aware return relative to the structural
  portfolio.
- Controller: trajectory-level policy objective plus step-level frozen
  counterfactual supervision.

Training uses a proportional cost of `0.00005` (0.005%) for both markets. This
is an intentional optimization choice: applying the complete 0.01% evaluation
cost during learning can dominate the weak day-level incremental reward and
prevent the Trader from discovering useful daily refinements. Paper evaluation
is nevertheless replayed at `0.0001` (0.01%).

## A.5 Progressive training procedure

Both NASDAQ-100 seed 49 and CSI-300 seed 90 use the same five stages:

1. Manager warm-up.
2. Trader warm-up with the Manager frozen.
3. Fixed-interval Manager–Trader stabilization.
4. Controller training with Manager and Trader frozen.
5. End-to-end alignment with all roles unfrozen.

The exact command arrays are supplied under `../checkpoints/`, and the executable
driver is `../scripts/train/run_five_stage_training.sh`.
Checkpoint hashes and the fee split are locked in `MODEL_VERSION.json`.
