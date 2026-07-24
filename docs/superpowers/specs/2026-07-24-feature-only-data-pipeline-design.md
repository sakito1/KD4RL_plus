# Feature-Only Data Pipeline Design

## Summary

The active CMTFlow and baseline workflows shall read market features from
`config.dataset["feature_path"]`. The obsolete SSM-only baseline, SSM runtime
inputs, rule-based SSM switching, and SSM generation pipeline are removed.
No compatibility path is retained for checkpoints created with the former SSM
controller interface.

## Scope

### CMTFlow

- Replace the SSM-oriented portfolio loader with a feature-only loader.
- Load aligned feature tensors, adjusted execution prices, dates, and stock
  mappings from `feature_path`.
- Derive missing adjusted-price and candle fields from raw OHLCV and
  `adjfactor` through the existing feature-standardization helper.
- Remove `h`, `z`, `p`, `q_bear`, and `q_bull` tensors from the environment.
- Remove the `ssm` and `held_p` entries from observations and rollout buffers.
- Change the learned Controller interface to consume only:
  - normalized asset feature history (`asset_state`);
  - drifted portfolio weights;
  - portfolio state;
  - the candidate switch portfolio.
- Remove the Controller fallback projection for SSM states.
- Remove `ssm_dim`, `ssm_data_path`, and market-specific SSM path overrides
  from the active training CLI and metadata.
- Remove the legacy rule-switch execution and evaluation path.

Outer Actor, Inner Actor, learned Controller objectives, transaction-cost
accounting, training phases, holding constraints, and test ablations remain
unchanged.

### Baselines

- Remove the SSM-only baseline from the baseline registry and execution order.
- Delete its implementation under `Baseline/SSM/`.
- AlphaStock reads `feature_path` directly and standardizes its required seven
  fields at load time.
- DeepTrader reads `feature_path` and creates its model-specific asset tensor
  from `adjopen`, `adjhigh`, `adjlow`, `adjclose`, `amount`, and `amp`, together
  with its return/split files, without SSM fields.
- DeepAries reads raw OHLCV-compatible fields from `feature_path`, then uses its
  native `YfinancePreprocessor` to generate its own normalized and rolling
  features.
- Classical price-based baselines continue to use `feature_path`.

### Obsolete SSM Code

- Delete the standalone `SSM_pipeline.py`.
- Remove active configuration fields and script arguments that refer to
  `feature_ssm` or SSM dimensions.
- Delete stale SSM-only tests or replace them with feature-only assertions when
  the test still covers an active behavior.
- Dataset `feature_ssm` directories are not restored.

## Data Contract

Each market supplies:

- one CSV per configured stock under `feature_path`;
- a `Date` column;
- either adjusted fields directly, or raw `open`, `high`, `low`, `close`,
  `volume`, and `adjfactor` sufficient to derive them.

The common trading calendar is the intersection across the configured stock
universe. Current validation establishes:

- Nasdaq-100: 39 stocks, 6,425 common dates;
- CSI-300: 53 stocks, 6,170 common dates.

Missing files, missing non-derivable fields, an empty common calendar, or
non-finite generated tensors must fail with a market/stock-specific error.

## Checkpoint Policy

Old SSM-interface checkpoints are intentionally unsupported. New CMTFlow
models must be trained from scratch under the feature-only pipeline. Checkpoint
loading remains strict for newly generated checkpoints.

## Tests

1. Feature-only loader succeeds for representative Nasdaq-100 and CSI-300
   files and returns finite feature/price arrays.
2. The loader derives all configured Nasdaq fields from OHLCV and `adjfactor`.
3. CMTFlow observations and rollout batches contain no `ssm` or `held_p`.
4. Controller forward/update paths require no SSM arguments.
5. End-to-end smoke training constructs the environment using `feature_path`.
6. Baseline registry contains no SSM-only baseline.
7. AlphaStock, DeepTrader, and DeepAries adapters never resolve
   `ssm_data_path`.
8. DeepTrader output matches its configured six input features.
9. DeepAries raw adapter output is processed by its native preprocessor.
10. Repository search over active code finds no runtime references to
    `feature_ssm`, `ssm_data_path`, `ssm_dim`, or the rule-switch path.

## Open Risks

- Old checkpoints cannot be reused.
- Nasdaq-100 gains additional dates present in `feature` but absent from the
  deleted `feature_ssm`; configured train/validation/test date bounds continue
  to control the experiment interval.
- Removing SSM parameters changes serialized model and optimizer layouts, so
  every reported feature-only result must come from a fresh run.
