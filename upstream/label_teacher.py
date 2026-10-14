"""Generate privileged Teacher labels used to build Student SFT data.

This module preserves the original six-field compact output protocol and the
local expansion rules used in the experiments. By default it labels the train
and validation prompt files. Change ``DEFAULT_DATASET`` in PyCharm or use the
optional command-line arguments.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DEFAULT_DATASET = "brn_all"
DEFAULT_SPLITS = ("train", "val")
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
SUPPORTED_SPLITS = ("train", "val", "test")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.7-max"
MAX_WORKERS = 4
MAX_RETRIES = 3
SHOW_PROGRESS = True

SYSTEM_PROMPT = """你是金融事件图谱标注器。根据事件、候选节点和关系类型，判断事件对每个候选节点的影响。
只输出JSON数组，不输出解释性文字。
每个候选输出[i,a,d,s,p,r]。
i=候选编号。
a=P维持/D弱化/I无关。
d=-1负向/0中性/1正向。
s=影响强度桶0-3。
p=置信度桶0-3。
r=30字以内原因。
必须覆盖所有候选编号，不得新增、不得遗漏。"""

ACTION_MAP = {
    "P": "PRESERVED",
    "D": "DECOUPLED",
    "I": "ISOLATED",
}
VALID_ACTION_CODES = {"P", "D", "I"}
VALID_DIRECTIONS = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E2T Teacher labels.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=SUPPORTED_SPLITS,
    )
    return parser.parse_args()


def dataset_dir(dataset: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / dataset


def split_paths(dataset: str, split: str) -> tuple[Path, Path]:
    root = dataset_dir(dataset)
    return (
        root / f"llm_exam_{split}_teacher_{dataset}.csv",
        root / f"llm_exam_{split}_teacher_{dataset}_answered.csv",
    )


def build_client():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
        if pd.isna(value):
            return default
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def clamp_bucket_0_3(value: object, default: int = 0) -> int:
    try:
        value = int(float(value))
    except Exception:
        return default
    return max(0, min(3, value))


def extract_json_array_robust(text: object) -> list | None:
    try:
        text = str(text).strip()
        left = text.find("[")
        right = text.rfind("]")
        if left != -1 and right != -1 and right > left:
            return json.loads(text[left : right + 1])
        match = re.search(r"\[.*\]", text, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception:
        return None


def parse_candidates(candidates_json: object) -> list[dict]:
    try:
        candidates = json.loads(str(candidates_json))
    except Exception:
        return []

    clean = []
    for item in candidates:
        candidate_id = safe_int(item.get("candidate_id"))
        target_code = str(item.get("target_code", "")).strip().zfill(6)
        target_name = str(item.get("target_name", "")).strip()
        relation_type = str(item.get("relation_type", "NULL")).strip()
        if candidate_id <= 0 or not target_code:
            continue
        clean.append(
            {
                "candidate_id": candidate_id,
                "target_code": target_code,
                "target_name": target_name,
                "relation_type": relation_type,
                "is_prior_neighbor": safe_int(item.get("is_prior_neighbor", 0)),
            }
        )
    return sorted(clean, key=lambda item: item["candidate_id"])


def normalize_one_compact_item(item: object) -> dict | None:
    if isinstance(item, list):
        if len(item) != 6:
            return None
        candidate_id = safe_int(item[0])
        action = str(item[1]).strip().upper()
        direction = safe_int(item[2])
        strength = clamp_bucket_0_3(item[3])
        confidence = clamp_bucket_0_3(item[4])
        reason = str(item[5]).strip()
    elif isinstance(item, dict):
        candidate_id = safe_int(item.get("i", item.get("candidate_id", 0)))
        action = str(item.get("a", item.get("action", ""))).strip().upper()
        direction = safe_int(item.get("d", item.get("impact_direction", 0)))
        strength = clamp_bucket_0_3(
            item.get("s", item.get("strength_bucket", 0))
        )
        confidence = clamp_bucket_0_3(
            item.get("p", item.get("confidence_bucket", 0))
        )
        reason = str(item.get("r", item.get("reason", ""))).strip()
    else:
        return None

    if action in {"PRESERVED", "KEEP", "PRESERVE"}:
        action = "P"
    elif action in {"DECOUPLED", "WEAKEN", "WEAKENED"}:
        action = "D"
    elif action in {"ISOLATED", "NULL", "IRRELEVANT"}:
        action = "I"

    if candidate_id <= 0 or action not in VALID_ACTION_CODES:
        return None
    if direction not in VALID_DIRECTIONS:
        return None
    if len(reason) > 30:
        reason = reason[:30]

    return {
        "i": candidate_id,
        "a": action,
        "d": direction,
        "s": strength,
        "p": confidence,
        "r": reason,
    }


def validate_and_expand_answer(
    parsed_json: object,
    candidates: list[dict],
    row: pd.Series,
) -> tuple[list[dict] | None, list[dict] | None]:
    if not isinstance(parsed_json, list) or not parsed_json:
        return None, None

    candidate_map = {
        int(item["candidate_id"]): item for item in candidates
    }
    expected_ids = set(candidate_map)
    compact_items = []
    for raw_item in parsed_json:
        item = normalize_one_compact_item(raw_item)
        if item is None or item["i"] not in expected_ids:
            return None, None
        compact_items.append(item)

    output_ids = [item["i"] for item in compact_items]
    if (
        len(output_ids) != len(expected_ids)
        or len(set(output_ids)) != len(output_ids)
        or set(output_ids) != expected_ids
    ):
        return None, None

    compact_items = sorted(compact_items, key=lambda item: item["i"])
    event_horizon = max(
        0,
        min(6, safe_int(row.get("Impact_Horizon", row.get("H", 0)))),
    )

    rich_items = []
    for item in compact_items:
        candidate = candidate_map[item["i"]]
        rich_items.append(
            {
                "candidate_id": item["i"],
                "target_code": candidate["target_code"],
                "target_name": candidate["target_name"],
                "relation_type": candidate["relation_type"],
                "topology_action": ACTION_MAP[item["a"]],
                "impact_direction": item["d"],
                "impact_magnitude": round(float(item["s"]) / 3.0, 4),
                "confidence": round(float(item["p"]) / 3.0, 4),
                "impact_horizon": event_horizon,
                "strength_bucket": item["s"],
                "confidence_bucket": item["p"],
                "reason": item["r"],
            }
        )
    return compact_items, rich_items


def dynamic_max_tokens(num_candidates: int) -> int:
    return min(640, max(256, 52 * max(1, int(num_candidates)) + 64))


def call_teacher(
    client,
    row: pd.Series,
) -> tuple[list[dict] | None, list[dict] | None]:
    candidates = parse_candidates(row["Candidates_JSON"])
    if not candidates:
        return None, None

    prompt = str(row.get("Teacher_Prompt", "")).strip()
    if not prompt:
        raise ValueError("Teacher_Prompt is empty.")

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.05, 0.20))
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01,
                top_p=0.01,
                max_tokens=dynamic_max_tokens(len(candidates)),
                seed=42,
                extra_body={"enable_thinking": False},
            )
            parsed = extract_json_array_robust(
                completion.choices[0].message.content.strip()
            )
            compact, rich = validate_and_expand_answer(parsed, candidates, row)
            if rich is not None:
                return compact, rich
            time.sleep(1.2 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None, None


def make_row_key(row: pd.Series | dict) -> str:
    return f"{row['Date']}__{str(row['Symbol']).zfill(6)}"


def load_existing_success(output_path: Path) -> dict[str, dict]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {}
    try:
        old = pd.read_csv(output_path, dtype={"Symbol": str})
        old["Symbol"] = old["Symbol"].astype(str).str.zfill(6)
        if "Annotate_Status" not in old.columns:
            return {}
        success = old[old["Annotate_Status"] == "SUCCESS"]
        return {
            make_row_key(row): row.to_dict()
            for _, row in success.iterrows()
        }
    except Exception:
        return {}


def get_output_columns(frame: pd.DataFrame) -> list[str]:
    optional = [
        "Z_Score",
        "Impact_Return",
        "Impact_Reaction_Date",
        "Impact_Direction",
        "Impact_Magnitude",
        "Impact_Confidence",
        "Impact_Horizon",
        "Impact_Strength_Bucket",
        "Impact_Confidence_Bucket",
    ]
    keep = [
        "Date",
        "Symbol",
        "Name",
        "Candidates_JSON",
        "Teacher_Compact_Answer",
        "Teacher_Answer",
        "Annotate_Status",
    ]
    return [column for column in optional if column in frame.columns] + keep


def append_row(row: dict, output_path: Path, output_columns: list[str]) -> None:
    frame = pd.DataFrame([row])
    for column in output_columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[output_columns]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    frame.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf_8_sig",
    )


def compact_output_file(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    frame = pd.read_csv(output_path, dtype={"Symbol": str})
    if frame.empty:
        return
    frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
    frame["_key"] = frame["Date"].astype(str) + "__" + frame["Symbol"]
    frame = (
        frame.drop_duplicates(subset=["_key"], keep="last")
        .drop(columns=["_key"])
        .sort_values(["Date", "Symbol"])
        .reset_index(drop=True)
    )
    frame.to_csv(output_path, index=False, encoding="utf_8_sig")


def label_split(
    client,
    dataset: str,
    split: str,
) -> tuple[Path, int, int]:
    input_path, output_path = split_paths(dataset, split)
    if not input_path.exists():
        raise FileNotFoundError(f"Teacher prompt file not found: {input_path}")

    frame = pd.read_csv(input_path, dtype={"Symbol": str})
    frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
    required = ["Date", "Symbol", "Name", "Candidates_JSON", "Teacher_Prompt"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{input_path} is missing columns: {missing}")

    output_columns = get_output_columns(frame)
    success_cache = load_existing_success(output_path)
    rows = [
        row.copy()
        for _, row in frame.iterrows()
        if make_row_key(row) not in success_cache
    ]

    success_count = 0
    fail_count = 0

    def process_row(row: pd.Series) -> pd.Series:
        row = row.copy()
        compact, rich = call_teacher(client, row)
        if rich is None:
            row["Teacher_Compact_Answer"] = ""
            row["Teacher_Answer"] = ""
            row["Annotate_Status"] = "FAILED"
        else:
            row["Teacher_Compact_Answer"] = json.dumps(
                compact,
                ensure_ascii=False,
            )
            row["Teacher_Answer"] = json.dumps(rich, ensure_ascii=False)
            row["Annotate_Status"] = "SUCCESS"
        return row

    if rows:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            futures = [executor.submit(process_row, row) for row in rows]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Teacher {split}",
                disable=not SHOW_PROGRESS,
            ):
                result = future.result()
                append_row(result.to_dict(), output_path, output_columns)
                if result.get("Annotate_Status") == "SUCCESS":
                    success_count += 1
                else:
                    fail_count += 1

    compact_output_file(output_path)
    return output_path, success_count, fail_count


def label_teacher(
    dataset: str = DEFAULT_DATASET,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> list[Path]:
    client = build_client()
    outputs = []
    summaries = []
    for split in splits:
        output, success, failed = label_split(client, dataset, split)
        outputs.append(output)
        summaries.append(f"{split}: +{success} success, {failed} failed")
    print(f"Teacher labels saved for {dataset} ({'; '.join(summaries)})")
    return outputs


def main() -> None:
    random.seed(42)
    args = parse_args()
    label_teacher(args.dataset, tuple(args.splits))


if __name__ == "__main__":
    main()
