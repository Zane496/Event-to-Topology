"""Download daily market features from BaoStock for a fixed stock universe.

The script can be executed directly in PyCharm using the default settings:

    python data/get_feature_baostoc.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Default settings
# =============================================================================

DEFAULT_DATASET = "cev"

DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2026-04-30"

DEFAULT_SLEEP_SECONDS = 0.2
DEFAULT_MAX_RETRIES = 3

SUPPORTED_DATASETS = ("cev", "inu", "emd", "zid")

QUERY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "tradestatus,pctChg"
)

OUTPUT_COLUMNS = [
    "Symbol",
    "Name",
    "Date",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Amount",
    "Amplitude",
    "Pct_Chg",
    "Is_Listed",
]


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments.

    All arguments have default values, so the script can be executed
    directly in PyCharm without configuring program parameters.
    """
    parser = argparse.ArgumentParser(
        description="Download BaoStock daily features for one E2T dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
        help=(
            "Dataset name. "
            f"Available choices: {', '.join(SUPPORTED_DATASETS)}. "
            f"Default: {DEFAULT_DATASET}."
        ),
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=(
            "Start date in YYYY-MM-DD format. "
            f"Default: {DEFAULT_START_DATE}."
        ),
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=(
            "End date in YYYY-MM-DD format. "
            f"Default: {DEFAULT_END_DATE}."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=(
            "Delay between stock requests in seconds. "
            f"Default: {DEFAULT_SLEEP_SECONDS}."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of requests for each stock. "
            f"Default: {DEFAULT_MAX_RETRIES}."
        ),
    )

    return parser.parse_args()


def resolve_paths(dataset: str) -> tuple[Path, Path]:
    """Resolve input and output paths for the selected dataset."""
    data_root = Path(__file__).resolve().parent
    dataset_dir = data_root / dataset

    nodes_path = dataset_dir / f"{dataset}_nodes.csv"
    features_path = dataset_dir / f"{dataset}_features.csv"

    return nodes_path, features_path


def validate_nodes(
    df_nodes: pd.DataFrame,
    nodes_path: Path,
) -> pd.DataFrame:
    """Validate and normalize the fixed stock-universe file."""
    required_columns = {"Symbol", "Name"}
    missing_columns = required_columns.difference(df_nodes.columns)

    if missing_columns:
        raise ValueError(
            f"{nodes_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df_nodes = df_nodes.copy()

    df_nodes["Symbol"] = (
        df_nodes["Symbol"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    df_nodes["Name"] = (
        df_nodes["Name"]
        .astype(str)
        .str.strip()
    )

    df_nodes = df_nodes[
        (df_nodes["Symbol"] != "")
        & (df_nodes["Name"] != "")
    ].copy()

    df_nodes = df_nodes.drop_duplicates(
        subset=["Symbol"],
        keep="first",
    ).reset_index(drop=True)

    if df_nodes.empty:
        raise ValueError(
            f"{nodes_path} contains no valid stock nodes."
        )

    return df_nodes


def to_baostock_code(symbol: str) -> str:
    """Convert a six-digit A-share code to BaoStock format."""
    if symbol.startswith(("6", "9")):
        exchange = "sh"
    else:
        exchange = "sz"

    return f"{exchange}.{symbol}"


def query_one_stock(
    symbol: str,
    name: str,
    start_date: str,
    end_date: str,
    max_retries: int,
) -> pd.DataFrame | None:
    """Download daily observations for one stock."""
    baostock_code = to_baostock_code(symbol)
    query_result = None

    for attempt in range(max_retries):
        query_result = bs.query_history_k_data_plus(
            baostock_code,
            QUERY_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1",
        )

        if query_result.error_code == "0":
            break

        if attempt + 1 < max_retries:
            time.sleep(1.0)

    if (
        query_result is None
        or query_result.error_code != "0"
    ):
        error_message = (
            query_result.error_msg
            if query_result is not None
            else "unknown error"
        )

        print(
            f"Warning: failed to download "
            f"{symbol} ({name}): {error_message}"
        )
        return None

    rows: list[list[str]] = []

    while query_result.next():
        rows.append(query_result.get_row_data())

    if not rows:
        print(
            f"Warning: no observations returned "
            f"for {symbol} ({name})."
        )
        return None

    df_stock = pd.DataFrame(
        rows,
        columns=query_result.fields,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pctChg",
        "tradestatus",
    ]

    for column in numeric_columns:
        df_stock[column] = pd.to_numeric(
            df_stock[column],
            errors="coerce",
        )

    df_stock["preclose"] = df_stock["preclose"].replace(
        0,
        np.nan,
    )

    df_stock["pctChg"] = (
        (df_stock["close"] - df_stock["preclose"])
        / df_stock["preclose"]
        * 100.0
    )

    df_stock["amplitude"] = (
        (df_stock["high"] - df_stock["low"])
        / df_stock["preclose"]
        * 100.0
    )

    df_stock["pctChg"] = (
        df_stock["pctChg"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df_stock["amplitude"] = (
        df_stock["amplitude"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df_stock["Symbol"] = symbol
    df_stock["Name"] = name

    return df_stock


def format_feature_panel(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine stock-level observations into one feature panel."""
    if not frames:
        raise RuntimeError(
            "No market observations were downloaded."
        )

    df_features = pd.concat(
        frames,
        ignore_index=True,
    )

    df_features = df_features.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
            "amount": "Amount",
            "pctChg": "Pct_Chg",
            "amplitude": "Amplitude",
            "tradestatus": "Is_Listed",
        }
    )

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in df_features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Downloaded feature panel is missing columns: "
            f"{missing_columns}"
        )

    df_features = df_features[OUTPUT_COLUMNS].copy()

    df_features["Date"] = pd.to_datetime(
        df_features["Date"],
        errors="coerce",
    )

    df_features = df_features.dropna(
        subset=["Date", "Symbol"]
    )

    decimal_places = {
        "Open": 2,
        "Close": 2,
        "High": 2,
        "Low": 2,
        "Volume": 0,
        "Amount": 0,
        "Amplitude": 2,
        "Pct_Chg": 2,
        "Is_Listed": 1,
    }

    df_features = df_features.round(decimal_places)

    df_features = (
        df_features
        .sort_values(["Date", "Symbol"])
        .reset_index(drop=True)
    )

    df_features["Date"] = df_features["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    return df_features


def download_features(
    dataset: str = DEFAULT_DATASET,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Path:
    """Download and save the market feature panel."""
    dataset = dataset.strip().lower()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Available datasets: {SUPPORTED_DATASETS}"
        )

    nodes_path, features_path = resolve_paths(dataset)

    print("=" * 72)
    print("BaoStock market feature collection")
    print(f"Dataset     : {dataset}")
    print(f"Node file   : {nodes_path}")
    print(f"Output file : {features_path}")
    print(f"Date range  : {start_date} to {end_date}")
    print("=" * 72)

    if not nodes_path.exists():
        raise FileNotFoundError(
            f"Node file not found: {nodes_path}\n"
            f"Expected path: "
            f"data/{dataset}/{dataset}_nodes.csv"
        )

    df_nodes = pd.read_csv(
        nodes_path,
        dtype={"Symbol": str},
    )

    df_nodes = validate_nodes(
        df_nodes,
        nodes_path,
    )

    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(
            f"BaoStock login failed: "
            f"{login_result.error_msg}"
        )

    feature_frames: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    try:
        node_iterator = df_nodes[
            ["Symbol", "Name"]
        ].itertuples(
            index=False,
            name=None,
        )

        for symbol, name in tqdm(
            node_iterator,
            total=len(df_nodes),
            desc=f"Downloading {dataset}",
        ):
            stock_frame = query_one_stock(
                symbol=symbol,
                name=name,
                start_date=start_date,
                end_date=end_date,
                max_retries=max(1, max_retries),
            )

            if stock_frame is None:
                failed_symbols.append(symbol)
            else:
                feature_frames.append(stock_frame)

            time.sleep(max(0.0, sleep_seconds))

    finally:
        bs.logout()

    df_features = format_feature_panel(
        feature_frames
    )

    features_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_features.to_csv(
        features_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print(f"Saved feature panel : {features_path}")
    print(f"Stocks requested    : {len(df_nodes)}")
    print(
        "Stocks downloaded   : "
        f"{len(df_nodes) - len(failed_symbols)}"
    )
    print(f"Panel rows           : {len(df_features)}")

    if failed_symbols:
        print(
            "Failed symbols       : "
            + ", ".join(failed_symbols)
        )

    print("=" * 72)

    return features_path


def main() -> None:
    args = parse_args()

    download_features(
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_seconds=args.sleep,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()