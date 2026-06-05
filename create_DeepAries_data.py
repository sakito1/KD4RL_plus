import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import utils.config as config


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_COLS = ["adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp", "body"]


def _to_timestamp(value):
    if value is None:
        return None
    return pd.to_datetime(value)


def infer_market_name():
    data_path = str(config.dataset.get("ssm_data_path", config.dataset.get("feature_path", ""))).lower()
    stock_path = str(config.dataset.get("stocks_path", "")).lower()
    if "沪深" in data_path or "sh" in stock_path:
        return "sh"
    return "nas"


def _read_stock_list(stocks_path):
    with open(stocks_path) as fh:
        return [line.strip() for line in fh if line.strip()]


def _standardize_feature_columns(df, csv_path):
    required_cols = list(config.dataset.get("features_name", DEFAULT_FEATURE_COLS))
    eps = 1e-12
    df = df.copy()

    if "adjfactor" in df.columns:
        adjfactor = pd.to_numeric(df["adjfactor"], errors="coerce").fillna(1.0)
    elif {"adjclose", "close"}.issubset(df.columns):
        close = pd.to_numeric(df["close"], errors="coerce")
        adjclose = pd.to_numeric(df["adjclose"], errors="coerce")
        adjfactor = (adjclose / (close + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        adjfactor = pd.Series(1.0, index=df.index)

    derived_specs = {
        "adjopen": ("open", adjfactor),
        "adjclose": ("close", adjfactor),
        "adjhigh": ("high", adjfactor),
        "adjlow": ("low", adjfactor),
    }
    for target, (source, factor) in derived_specs.items():
        if target not in df.columns and source in df.columns:
            df[target] = pd.to_numeric(df[source], errors="coerce") * factor

    if "amount" not in df.columns:
        if {"close", "volume"}.issubset(df.columns):
            df["amount"] = (
                pd.to_numeric(df["close"], errors="coerce")
                * pd.to_numeric(df["volume"], errors="coerce")
            )
        elif "volume" in df.columns:
            df["amount"] = pd.to_numeric(df["volume"], errors="coerce")

    if "amp" not in df.columns or "body" not in df.columns:
        if {"adjopen", "adjclose", "adjhigh", "adjlow"}.issubset(df.columns):
            day_range = (
                pd.to_numeric(df["adjhigh"], errors="coerce")
                - pd.to_numeric(df["adjlow"], errors="coerce")
                + eps
            )
            if "amp" not in df.columns:
                df["amp"] = day_range / (pd.to_numeric(df["adjopen"], errors="coerce") + eps)
            if "body" not in df.columns:
                df["body"] = (
                    pd.to_numeric(df["adjclose"], errors="coerce")
                    - pd.to_numeric(df["adjopen"], errors="coerce")
                ) / day_range

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} missing columns: {missing_cols}")

    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    return df[required_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_deeparies_dataframe(
    feature_path=None,
    stocks_path=None,
    start_date=None,
    end_date=None,
    stocks_limit=None,
):
    feature_path = Path(feature_path or config.dataset.get("ssm_data_path", config.dataset["feature_path"]))
    stocks_path = stocks_path or config.dataset["stocks_path"]
    start_date = _to_timestamp(start_date or config.train_start_date)
    end_date = _to_timestamp(end_date or config.test_end_date)
    stocks = _read_stock_list(stocks_path)
    if stocks_limit is not None:
        stocks = stocks[: int(stocks_limit)]

    required_cols = list(config.dataset.get("features_name", DEFAULT_FEATURE_COLS))
    frames = []
    missing = []
    for stock in stocks:
        csv_path = feature_path / f"{stock}.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            raise ValueError(f"{csv_path} missing Date column")
        feature_df = _standardize_feature_columns(df, csv_path)
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        df["tic"] = stock
        df[required_cols] = feature_df.loc[df.index, required_cols]
        frames.append(df[["date", "tic", *required_cols]])

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} stock csv files, first entries:\n{preview}")
    if not frames:
        raise ValueError("No stock data was loaded for DeepAries.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)

    # DeepAries assumes every date contains the same asset universe.
    expected = data["tic"].nunique()
    complete_dates = data.groupby("date")["tic"].nunique()
    complete_dates = complete_dates[complete_dates == expected].index
    data = data[data["date"].isin(complete_dates)].copy()
    if data.empty:
        raise ValueError("No complete dates remain after aligning the DeepAries asset universe.")
    return data


def save_deeparies_data(
    market=None,
    output_root=None,
    data_type="general",
    feature_path=None,
    stocks_path=None,
    start_date=None,
    end_date=None,
    stocks_limit=None,
):
    market = market or infer_market_name()
    output_root = Path(output_root or ROOT / "DeepAries" / "data" / market)
    source_feature_path = Path(feature_path or config.dataset.get("ssm_data_path", config.dataset["feature_path"]))
    feature_cols = list(config.dataset.get("features_name", DEFAULT_FEATURE_COLS))
    output_root.mkdir(parents=True, exist_ok=True)
    data = build_deeparies_dataframe(
        feature_path=source_feature_path,
        stocks_path=stocks_path,
        start_date=start_date,
        end_date=end_date,
        stocks_limit=stocks_limit,
    )

    raw_path = output_root / f"{market}_data.csv"
    processed_path = output_root / f"{market}_{data_type}_data.csv"
    data.to_csv(raw_path, index=False)
    data.to_csv(processed_path, index=False)

    summary = {
        "market": market,
        "rows": int(len(data)),
        "stocks": int(data["tic"].nunique()),
        "dates": int(data["date"].nunique()),
        "start": str(data["date"].min().date()),
        "end": str(data["date"].max().date()),
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "feature_path": str(source_feature_path),
        "feature_cols": feature_cols,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export KD4RL feature data for DeepAries.")
    parser.add_argument("--market", default=None, help="DeepAries market name, e.g. nas or sh.")
    parser.add_argument("--output_root", default=None, help="Directory that receives DeepAries CSV files.")
    parser.add_argument("--data_type", default="general", help="DeepAries data type suffix.")
    parser.add_argument("--feature_path", default=None, help="Override config.dataset['ssm_data_path'].")
    parser.add_argument("--stocks_path", default=None, help="Override config.dataset['stocks_path'].")
    parser.add_argument("--start_date", default=None, help="Default: config.train_start_date.")
    parser.add_argument("--end_date", default=None, help="Default: config.test_end_date.")
    parser.add_argument("--stocks_limit", type=int, default=None, help="Optional small asset subset.")
    args = parser.parse_args()

    summary = save_deeparies_data(**vars(args))
    print("DeepAries data saved:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
