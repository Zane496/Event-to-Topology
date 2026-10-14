"""Split the E2T event benchmark by absolute calendar time."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
TRAIN_END_DATE = pd.Timestamp("2024-12-31")
VAL_END_DATE = pd.Timestamp("2025-06-30")

REQUIRED_COLUMNS = [
    "Date",
    "Symbol",
    "Name",
    "Z_Score",
    "Impact_Return",
    "Impact_Reaction_Date",
    "Impact_Direction",
    "Impact_Magnitude",
    "Impact_Confidence",
    "Impact_Horizon",
    "Impact_Strength_Bucket",
    "Impact_Confidence_Bucket",
    "Teacher_Input",
    "Student_Input",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split E2T events by time.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def resolve_paths(dataset: str) -> tuple[Path, Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / dataset

    input_file = data_dir / f"llm_all_events_{dataset}.csv"

    return (
        input_file,
        data_dir / f"events_train_2022_2024_{dataset}.csv",
        data_dir / f"events_val_2025H1_{dataset}.csv",
        data_dir / f"events_test_2025H2_2026_{dataset}.csv",
    )


def split_events(dataset: str = DEFAULT_DATASET) -> tuple[Path, Path, Path]:
    input_file, train_file, val_file, test_file = resolve_paths(dataset)
    if not input_file.exists():
        raise FileNotFoundError(f"All-announcement events not found: {input_file}. Run Step 1 first.")

    events = pd.read_csv(input_file, dtype={"Symbol": str})

    # Preserve legacy Chinese column aliases because some frozen experiment files
    # were generated with these headers.
    rename_map = {}
    if "日期" in events.columns and "Date" not in events.columns:
        rename_map["日期"] = "Date"
    if "代码" in events.columns and "Symbol" not in events.columns:
        rename_map["代码"] = "Symbol"
    if "名称" in events.columns and "Name" not in events.columns:
        rename_map["名称"] = "Name"
    if rename_map:
        events = events.rename(columns=rename_map)

    missing = [column for column in REQUIRED_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError(f"Event file is missing columns: {missing}")

    events["Symbol"] = events["Symbol"].astype(str).str.zfill(6)
    events["Date"] = pd.to_datetime(events["Date"])
    events["Impact_Reaction_Date"] = pd.to_datetime(events["Impact_Reaction_Date"])

    numeric_columns = [
        "Z_Score",
        "Impact_Return",
        "Impact_Direction",
        "Impact_Magnitude",
        "Impact_Confidence",
        "Impact_Horizon",
        "Impact_Strength_Bucket",
        "Impact_Confidence_Bucket",
    ]
    for column in numeric_columns:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    events = events.dropna(
        subset=[
            "Date",
            "Symbol",
            "Z_Score",
            "Impact_Direction",
            "Impact_Magnitude",
            "Impact_Confidence",
            "Teacher_Input",
            "Student_Input",
        ]
    ).copy()
    events = events.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    train = events[events["Date"] <= TRAIN_END_DATE].copy()
    val = events[
        (events["Date"] > TRAIN_END_DATE)
        & (events["Date"] <= VAL_END_DATE)
    ].copy()
    test = events[events["Date"] > VAL_END_DATE].copy()

    for frame in (train, val, test):
        if not frame.empty:
            frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
            frame["Impact_Reaction_Date"] = frame[
                "Impact_Reaction_Date"
            ].dt.strftime("%Y-%m-%d")

    train_file.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_file, index=False, encoding="utf_8_sig")
    val.to_csv(val_file, index=False, encoding="utf_8_sig")
    test.to_csv(test_file, index=False, encoding="utf_8_sig")

    print(
        f"Saved train/val/test events: {len(train)}/{len(val)}/{len(test)} "
        f"to {train_file.parent}"
    )
    return train_file, val_file, test_file


def main() -> None:
    args = parse_args()
    split_events(args.dataset)


if __name__ == "__main__":
    main()
