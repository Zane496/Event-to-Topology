"""Build full-announcement event datasets for E2T.

The calculations, thresholds, temporal alignment, and Teacher/Student input
separation follow the original experimental script. PyCharm can run the file
directly with ``DEFAULT_DATASET``.
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from tqdm import tqdm

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
GLOBAL_START = "20220101"
GLOBAL_END = "20260430"
POS_Z_THRESHOLD = 1.64
NEG_Z_THRESHOLD = -1.28
SHOW_PROGRESS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E2T event datasets.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def resolve_paths(dataset: str) -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / dataset
    return {
        "features": data_dir / f"{dataset}_features.csv",
        "all_events": data_dir / f"llm_all_events_{dataset}.csv",
        "forecast_cache": data_dir / "forecast_cache_20220101_20260430.csv",
            }


def smart_clean_and_truncate(text: object, max_len: int = 500) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text.replace("\n", " ").replace("\r", " "),
    ).strip()
    boilerplates = [
        r"根据《.*?准则》.*?的相关规定，?",
        r"基于谨慎性原则，?",
        r"经公司财务部门.*?初步测算，?",
        r"敬请广大投资者谨慎决策，?注意投资风险。?",
        r"本次业绩预告是公司财务部门初步测算的结果，?",
        r"未经注册会计师审计。?",
    ]
    for pattern in boilerplates:
        text = re.sub(pattern, "", text)

    if len(text) > max_len:
        cut = text[:max_len]
        match = re.search(r"[。！？；，](?=[^。！？；，]*$)", cut)
        text = cut[: match.end()] if match else cut
        text += "..."
    return text


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def clip01(value: object) -> float:
    return float(np.clip(safe_float(value), 0.0, 1.0))


def bucketize_0_3(value: object) -> int:
    value = clip01(value)
    return int(np.clip(round(value * 3), 0, 3))


def compute_impact_labels(
    final_z: object,
    final_vol: object,
    max_z_idx: int,
) -> dict[str, float | int]:
    final_z = safe_float(final_z)
    final_vol = safe_float(final_vol)

    if final_z > POS_Z_THRESHOLD:
        direction = 1
    elif final_z < NEG_Z_THRESHOLD:
        direction = -1
    else:
        direction = 0

    magnitude = clip01(abs(final_z) / 3.0)
    volume_strength = clip01(final_vol / 3.0)
    confidence = clip01(0.60 * magnitude + 0.40 * volume_strength)

    return {
        "Impact_Direction": direction,
        "Impact_Magnitude": magnitude,
        "Impact_Confidence": confidence,
        "Impact_Horizon": int(max_z_idx + 1),
        "Impact_Strength_Bucket": bucketize_0_3(magnitude),
        "Impact_Confidence_Bucket": bucketize_0_3(confidence),
    }


def build_pre_market_info(pre_data: dict) -> str:
    ret_mean = safe_float(pre_data.get("Ret_Mean_T1"))
    ret_std = safe_float(pre_data.get("Ret_Std_T1"))
    ret_5d = safe_float(pre_data.get("Ret_5D_Mean_T1"))
    volume = safe_float(pre_data.get("Volume"))
    volume_mean = safe_float(pre_data.get("Vol_Mean_T1"))
    volume_ratio = 0.0 if volume_mean <= 0 else volume / volume_mean

    return (
        f"[事前市场状态] 公告前20日日均涨跌幅{ret_mean:.2f}%，"
        f"波动率{ret_std:.2f}%，"
        f"近5日日均涨跌幅{ret_5d:.2f}%，"
        f"成交量为近20日均值的{volume_ratio:.1f}倍。"
    )


def load_or_fetch_forecasts(
    symbols: list[str] | np.ndarray,
    cache_file: Path,
) -> pd.DataFrame:
    if cache_file.exists():
        return pd.read_csv(cache_file, dtype={"ts_code": str})

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "Set TUSHARE_TOKEN or provide the existing forecast cache file."
        )

    ts.set_token(token)
    pro = ts.pro_api()
    forecasts: list[pd.DataFrame] = []

    for symbol in tqdm(
        symbols,
        desc="Forecasts",
        disable=not SHOW_PROGRESS,
    ):
        ts_code = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
        for attempt in range(3):
            try:
                frame = pro.forecast(
                    ts_code=ts_code,
                    start_date=GLOBAL_START,
                    end_date=GLOBAL_END,
                )
                if not frame.empty:
                    forecasts.append(frame)
                break
            except Exception:
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))
        time.sleep(0.3)

    if not forecasts:
        return pd.DataFrame()

    result = pd.concat(forecasts, ignore_index=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_file, index=False, encoding="utf_8_sig")
    return result


def build_events(dataset: str = DEFAULT_DATASET) -> Path:
    paths = resolve_paths(dataset)
    feature_file = paths["features"]
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df_features = pd.read_csv(feature_file, dtype={"Symbol": str})
    required_features = {"Symbol", "Name", "Date", "Pct_Chg", "Volume", "Is_Listed"}
    missing = required_features.difference(df_features.columns)
    if missing:
        raise ValueError(f"Feature file is missing columns: {sorted(missing)}")

    df_features["Symbol"] = df_features["Symbol"].astype(str).str.zfill(6)
    df_features["Date"] = pd.to_datetime(df_features["Date"])
    df_features = (
        df_features.drop_duplicates(subset=["Symbol", "Date"], keep="last")
        .sort_values(["Symbol", "Date"])
        .reset_index(drop=True)
    )

    df_features.loc[
        df_features["Is_Listed"] == 0,
        ["Pct_Chg", "Volume"],
    ] = np.nan
    df_features["Ret_Mean_T1"] = df_features.groupby("Symbol")["Pct_Chg"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).mean()
    )
    df_features["Ret_Std_T1"] = df_features.groupby("Symbol")["Pct_Chg"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).std()
    )
    df_features["Vol_Mean_T1"] = df_features.groupby("Symbol")["Volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).mean()
    )
    df_features["Ret_5D_Mean_T1"] = df_features.groupby("Symbol")["Pct_Chg"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=3).mean()
    )

    symbols = df_features["Symbol"].unique()
    df_forecasts = load_or_fetch_forecasts(symbols, paths["forecast_cache"])
    if df_forecasts.empty:
        raise RuntimeError("No forecast announcements were retrieved.")

    df_forecasts["ann_date"] = pd.to_datetime(df_forecasts["ann_date"])
    df_forecasts = (
        df_forecasts.dropna(subset=["change_reason"])
        .drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
    )

    feature_map = df_features.set_index(["Symbol", "Date"]).to_dict("index")
    valid_trading = df_features[df_features["Is_Listed"] == 1]
    trading_dates = (
        valid_trading.groupby("Symbol")["Date"]
        .apply(lambda values: sorted(list(values)))
        .to_dict()
    )

    rows: list[dict] = []
    for _, event in tqdm(
        df_forecasts.iterrows(),
        total=len(df_forecasts),
        desc="Events",
        disable=not SHOW_PROGRESS,
    ):
        symbol = str(event["ts_code"])[:6].zfill(6)
        announcement_date = event["ann_date"]

        try:
            valid_dates = trading_dates.get(symbol, [])
            if not valid_dates:
                continue

            index = bisect.bisect_left(valid_dates, announcement_date)
            if index >= len(valid_dates):
                continue

            target_days = valid_dates[index : min(index + 3, len(valid_dates))]
            z_scores: list[float] = []
            volume_ratios: list[float] = []
            returns: list[float] = []

            for trading_date in target_days:
                day = feature_map.get((symbol, trading_date))
                if not day:
                    continue

                ret_std = safe_float(day.get("Ret_Std_T1"))
                ret_mean = safe_float(day.get("Ret_Mean_T1"))
                pct_chg = safe_float(day.get("Pct_Chg"))
                volume = safe_float(day.get("Volume"))
                volume_mean = safe_float(day.get("Vol_Mean_T1"))
                if ret_std <= 0 or volume_mean <= 0:
                    continue

                z_scores.append((pct_chg - ret_mean) / ret_std)
                volume_ratios.append(volume / volume_mean)
                returns.append(pct_chg)

            if not z_scores:
                continue

            max_index = int(np.argmax(np.abs(z_scores)))
            final_z = safe_float(z_scores[max_index])
            final_volume = safe_float(volume_ratios[max_index])
            final_return = safe_float(returns[max_index])
            reaction_date = target_days[max_index].strftime("%Y-%m-%d")
            labels = compute_impact_labels(final_z, final_volume, max_index)

            if final_z > POS_Z_THRESHOLD:
                trend = "强力向上重估"
            elif final_z < NEG_Z_THRESHOLD:
                trend = "向下破位"
            else:
                trend = "情绪震荡"

            volume_description = (
                f"放大 {final_volume:.1f} 倍"
                if final_volume >= 1.0
                else f"萎缩至均值的 {final_volume:.1f} 倍"
            )

            previous_index = index - 1
            if previous_index < 0:
                continue
            previous_date = valid_dates[previous_index]
            pre_data = feature_map.get((symbol, previous_date))
            if not pre_data or pd.isna(pre_data.get("Ret_Mean_T1")):
                continue

            pre_market_info = build_pre_market_info(pre_data)
            stock_name = pre_data.get("Name", symbol)
            reason = smart_clean_and_truncate(event["change_reason"])
            fundamental_text = f"[基本面异动] {reason}"
            post_market_desc = (
                f"[事后市场反馈] 异动日 {reaction_date} 录得核心冲击："
                f"涨跌幅 {final_return:.2f}% "
                f"(真实Z-Score: {final_z:.2f})，"
                f"成交量{volume_description}。市场呈现{trend}。"
            )
            privileged_label_desc = (
                "[教师特权结构化标签] "
                f"Direction={labels['Impact_Direction']}; "
                f"Magnitude={labels['Impact_Magnitude']:.3f}; "
                f"Confidence={labels['Impact_Confidence']:.3f}; "
                f"Horizon={labels['Impact_Horizon']}; "
                f"StrengthBucket={labels['Impact_Strength_Bucket']}; "
                f"ConfidenceBucket={labels['Impact_Confidence_Bucket']}."
            )

            rows.append(
                {
                    "Date": announcement_date.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "Name": stock_name,
                    "Z_Score": final_z,
                    "Impact_Return": final_return,
                    "Impact_Reaction_Date": reaction_date,
                    "Impact_Direction": labels["Impact_Direction"],
                    "Impact_Magnitude": labels["Impact_Magnitude"],
                    "Impact_Confidence": labels["Impact_Confidence"],
                    "Impact_Horizon": labels["Impact_Horizon"],
                    "Impact_Strength_Bucket": labels["Impact_Strength_Bucket"],
                    "Impact_Confidence_Bucket": labels["Impact_Confidence_Bucket"],
                    "Teacher_Input": (
                        f"{pre_market_info}\n{post_market_desc}\n"
                        f"{privileged_label_desc}\n{fundamental_text}"
                    ),
                    "Student_Input": f"{pre_market_info}\n{fundamental_text}",
                }
            )
        except Exception:
            continue

    df_final = pd.DataFrame(rows)
    if df_final.empty:
        raise RuntimeError("No valid events matched the market feature panel.")

    df_all = df_final.copy()
    df_all["Event_Setting"] = "full_event"
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    df_all.to_csv(paths["all_events"], index=False, encoding="utf_8_sig")
    print(f"[Step 1] Saved {len(df_all)} all-announcement events to {paths['all_events']}")
    return paths["all_events"]


def main() -> None:
    args = parse_args()
    build_events(args.dataset)


if __name__ == "__main__":
    main()
