"""Generate validation/test Teacher gold labels and edge-level references.

The default mode only expands an existing answered file, matching the original
Step 6 entry point. Use ``--annotate`` to call the privileged Teacher first.
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
DEFAULT_SPLIT = "val"
DEFAULT_ANNOTATE = False
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
SUPPORTED_SPLITS = ("val", "test")
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

ACTION_MAP = {"P": "PRESERVED", "D": "DECOUPLED", "I": "ISOLATED"}
ACTION_REVERSE_MAP = {value: key for key, value in ACTION_MAP.items()}
VALID_ACTION_CODES = {"P", "D", "I"}
VALID_DIRECTIONS = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E2T Teacher gold.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=SUPPORTED_SPLITS,
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=DEFAULT_ANNOTATE,
        help="Call the Teacher before building edge-level gold.",
    )
    return parser.parse_args()


def resolve_paths(dataset: str, split: str) -> tuple[Path, Path, Path]:
    data_dir = Path(__file__).resolve().parents[1] / "data" / dataset
    return (
        data_dir / f"llm_exam_{split}_teacher_{dataset}.csv",
        data_dir / f"llm_exam_{split}_teacher_{dataset}_answered.csv",
        data_dir / f"qwen_max_{split}_gold_edges_{dataset}.csv",
    )


def build_client():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or QWEN_API_KEY.")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp_bucket_0_3(value: object, default: int = 0) -> int:
    try:
        value = int(float(value))
    except Exception:
        return default
    return max(0, min(3, value))


def continuous_to_bucket_0_3(value: object, default: int = 0) -> int:
    try:
        value = float(value)
    except Exception:
        return default
    if pd.isna(value):
        return default
    value = max(0.0, min(1.0, value))
    return int(max(0, min(3, round(value * 3))))


def bucket_to_value(bucket: int) -> float:
    return {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}[
        int(max(0, min(3, bucket)))
    ]


def parse_json_safe(value: object) -> object | None:
    try:
        value = str(value).strip()
        return json.loads(value) if value else None
    except Exception:
        return None


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
    candidates = parse_json_safe(candidates_json)
    if not isinstance(candidates, list):
        return []

    clean = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        candidate_id = safe_int(item.get("candidate_id"))
        code = str(item.get("target_code", "")).strip().zfill(6)
        if candidate_id <= 0 or not code:
            continue
        clean.append(
            {
                "candidate_id": candidate_id,
                "target_code": code,
                "target_name": str(item.get("target_name", "")).strip(),
                "relation_type": str(item.get("relation_type", "NULL")).strip(),
                "is_prior_neighbor": safe_int(item.get("is_prior_neighbor", 0)),
            }
        )
    return sorted(clean, key=lambda item: item["candidate_id"])


def normalize_action_code(value: object) -> str | None:
    value = str(value).strip().upper()
    if value in {"P", "PRESERVED", "KEEP"}:
        return "P"
    if value in {"D", "DECOUPLED", "WEAKEN", "WEAKENED"}:
        return "D"
    if value in {"I", "ISOLATED", "NULL", "NONE"}:
        return "I"
    return None


def normalize_one_item(item: object) -> dict | None:
    if isinstance(item, list):
        if len(item) != 6:
            return None
        candidate_id = safe_int(item[0])
        action = normalize_action_code(item[1])
        direction = safe_int(item[2])
        strength = clamp_bucket_0_3(item[3])
        confidence = clamp_bucket_0_3(item[4])
        reason = str(item[5]).strip()
    elif isinstance(item, dict):
        candidate_id = safe_int(item.get("i", item.get("candidate_id", 0)))
        action = normalize_action_code(
            item.get("a", item.get("topology_action", item.get("action", "")))
        )
        direction = safe_int(item.get("d", item.get("impact_direction", 0)))
        if "s" in item:
            strength = clamp_bucket_0_3(item.get("s"))
        elif "strength_bucket" in item:
            strength = clamp_bucket_0_3(item.get("strength_bucket"))
        else:
            strength = continuous_to_bucket_0_3(
                item.get("impact_magnitude", 0.0)
            )
        if "p" in item:
            confidence = clamp_bucket_0_3(item.get("p"))
        elif "confidence_bucket" in item:
            confidence = clamp_bucket_0_3(item.get("confidence_bucket"))
        else:
            confidence = continuous_to_bucket_0_3(item.get("confidence", 0.0))
        reason = str(item.get("r", item.get("reason", ""))).strip()
    else:
        return None

    if action not in VALID_ACTION_CODES:
        return None
    if direction not in VALID_DIRECTIONS:
        direction = 0
    reason = reason.replace("\n", " ").replace("\r", " ").strip()
    if len(reason) > 40:
        reason = reason[:40]
    return {
        "candidate_id": candidate_id,
        "action_code": action,
        "direction": direction,
        "strength_bucket": strength,
        "confidence_bucket": confidence,
        "reason": reason,
    }


def validate_and_expand_answer(
    parsed_json: object,
    candidates: list[dict],
) -> tuple[list[list] | None, list[dict] | None]:
    if not isinstance(parsed_json, list) or not parsed_json:
        return None, None

    candidate_map = {
        int(item["candidate_id"]): item for item in candidates
    }
    expected_ids = set(candidate_map)
    normalized = []
    for raw_item in parsed_json:
        item = normalize_one_item(raw_item)
        if item is None or item["candidate_id"] not in expected_ids:
            return None, None
        normalized.append(item)

    output_ids = [item["candidate_id"] for item in normalized]
    if (
        len(output_ids) != len(expected_ids)
        or len(set(output_ids)) != len(output_ids)
        or set(output_ids) != expected_ids
    ):
        return None, None

    compact = []
    rich = []
    for item in sorted(normalized, key=lambda row: row["candidate_id"]):
        candidate = candidate_map[item["candidate_id"]]
        compact.append(
            [
                item["candidate_id"],
                item["action_code"],
                item["direction"],
                item["strength_bucket"],
                item["confidence_bucket"],
                item["reason"],
            ]
        )
        rich.append(
            {
                "candidate_id": item["candidate_id"],
                "target_code": candidate["target_code"],
                "target_name": candidate["target_name"],
                "relation_type": candidate["relation_type"],
                "is_prior_neighbor": candidate["is_prior_neighbor"],
                "topology_action": ACTION_MAP[item["action_code"]],
                "impact_direction": item["direction"],
                "strength_bucket": item["strength_bucket"],
                "confidence_bucket": item["confidence_bucket"],
                "impact_magnitude": round(
                    bucket_to_value(item["strength_bucket"]), 4
                ),
                "confidence": round(
                    bucket_to_value(item["confidence_bucket"]), 4
                ),
                "reason": item["reason"],
            }
        )
    return compact, rich


def dynamic_max_tokens(num_candidates: int) -> int:
    return min(640, max(256, 52 * max(1, int(num_candidates)) + 64))


def call_teacher(
    client,
    row: pd.Series,
) -> tuple[list[list] | None, list[dict] | None]:
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
            compact, rich = validate_and_expand_answer(parsed, candidates)
            if rich is not None:
                return compact, rich
            time.sleep(1.2 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None, None


def event_key(row: pd.Series | dict) -> str:
    return f"{row['Date']}__{str(row['Symbol']).zfill(6)}"


def output_columns(frame: pd.DataFrame) -> list[str]:
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


def load_existing_success(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(path, dtype={"Symbol": str})
        if frame.empty or "Annotate_Status" not in frame.columns:
            return set()
        frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
        return {
            event_key(row)
            for _, row in frame[frame["Annotate_Status"] == "SUCCESS"].iterrows()
        }
    except Exception:
        return set()


def append_row(row: dict, path: Path, columns: list[str]) -> None:
    frame = pd.DataFrame([row])
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[columns]
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
        encoding="utf_8_sig",
    )


def compact_answered_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    frame = pd.read_csv(path, dtype={"Symbol": str})
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
    frame.to_csv(path, index=False, encoding="utf_8_sig")


def annotate_teacher_file(input_file: Path, answered_file: Path) -> tuple[int, int]:
    if not input_file.exists():
        raise FileNotFoundError(f"Teacher prompt file not found: {input_file}")
    client = build_client()
    frame = pd.read_csv(input_file, dtype={"Symbol": str})
    frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
    required = ["Date", "Symbol", "Name", "Candidates_JSON", "Teacher_Prompt"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{input_file} is missing columns: {missing}")

    columns = output_columns(frame)
    completed = load_existing_success(answered_file)
    rows = [
        row.copy()
        for _, row in frame.iterrows()
        if event_key(row) not in completed
    ]

    def process(row: pd.Series) -> pd.Series:
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

    success = 0
    failed = 0
    if rows:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            futures = [executor.submit(process, row) for row in rows]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Teacher gold",
                disable=not SHOW_PROGRESS,
            ):
                result = future.result()
                append_row(result.to_dict(), answered_file, columns)
                if result.get("Annotate_Status") == "SUCCESS":
                    success += 1
                else:
                    failed += 1
    compact_answered_file(answered_file)
    return success, failed


def build_edge_gold(answered_file: Path, gold_file: Path) -> int:
    if not answered_file.exists():
        raise FileNotFoundError(f"Answered Teacher file not found: {answered_file}")
    frame = pd.read_csv(answered_file, dtype={"Symbol": str})
    if frame.empty:
        raise RuntimeError(f"Answered Teacher file is empty: {answered_file}")
    frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
    if "Annotate_Status" not in frame.columns:
        raise ValueError("Answered Teacher file is missing Annotate_Status.")
    frame = frame[frame["Annotate_Status"] == "SUCCESS"].copy()

    rows = []
    for _, event in frame.iterrows():
        items = parse_json_safe(event.get("Teacher_Answer", ""))
        if not isinstance(items, list):
            continue
        for item in items:
            action = str(item.get("topology_action", "UNKNOWN")).strip()
            rows.append(
                {
                    "Date": event["Date"],
                    "Symbol": str(event["Symbol"]).zfill(6),
                    "Name": event.get("Name", ""),
                    "Candidate_ID": safe_int(item.get("candidate_id")),
                    "Target_Code": str(item.get("target_code", "")).zfill(6),
                    "Target_Name": item.get("target_name", ""),
                    "Relation_Type": item.get("relation_type", "NULL"),
                    "Is_Prior_Neighbor": safe_int(
                        item.get("is_prior_neighbor", 0)
                    ),
                    "True_Action": action,
                    "True_Action_Code": ACTION_REVERSE_MAP.get(action, "UNK"),
                    "True_Direction": safe_int(item.get("impact_direction", 0)),
                    "True_Strength_Bucket": clamp_bucket_0_3(
                        item.get("strength_bucket", 0)
                    ),
                    "True_Confidence_Bucket": clamp_bucket_0_3(
                        item.get("confidence_bucket", 0)
                    ),
                    "True_Magnitude": safe_float(
                        item.get("impact_magnitude", 0.0)
                    ),
                    "True_Confidence": safe_float(item.get("confidence", 0.0)),
                    "True_Reason": item.get("reason", ""),
                }
            )

    gold = pd.DataFrame(rows)
    if gold.empty:
        raise RuntimeError("No edge-level gold rows were generated.")
    gold = gold.sort_values(
        ["Date", "Symbol", "Candidate_ID", "Target_Code"]
    ).reset_index(drop=True)
    gold.to_csv(gold_file, index=False, encoding="utf_8_sig")
    return len(gold)


def generate_gold(
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    annotate: bool = DEFAULT_ANNOTATE,
) -> Path:
    input_file, answered_file, gold_file = resolve_paths(dataset, split)
    success = failed = 0
    if annotate:
        success, failed = annotate_teacher_file(input_file, answered_file)
    edge_count = build_edge_gold(answered_file, gold_file)
    suffix = f", +{success} success/{failed} failed" if annotate else ""
    print(f"Saved {edge_count} {split} gold edges to {gold_file}{suffix}")
    return gold_file


def main() -> None:
    random.seed(42)
    args = parse_args()
    generate_gold(args.dataset, args.split, args.annotate)


if __name__ == "__main__":
    main()
