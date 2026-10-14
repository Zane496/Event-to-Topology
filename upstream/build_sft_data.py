"""Pair Student prompts with privileged Teacher labels for compact SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
SAVE_AUDIT = True

ACTION_TO_CODE = {
    "PRESERVED": "P",
    "DECOUPLED": "D",
    "ISOLATED": "I",
    "P": "P",
    "D": "D",
    "I": "I",
}
VALID_ACTION_CODES = {"P", "D", "I"}
VALID_DIRECTIONS = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact E2T SFT data.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def dataset_dir(dataset: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / dataset


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
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


def parse_json_safe(value: object) -> object | None:
    try:
        value = str(value).strip()
        return json.loads(value) if value else None
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
        name = str(item.get("target_name", "")).strip()
        relation = str(item.get("relation_type", "NULL")).strip()
        if candidate_id <= 0 or not code:
            continue
        clean.append(
            {
                "candidate_id": candidate_id,
                "target_code": code,
                "target_name": name,
                "relation_type": relation,
            }
        )
    return sorted(clean, key=lambda item: item["candidate_id"])


def normalize_action_to_code(action: object) -> str | None:
    return ACTION_TO_CODE.get(str(action).strip().upper())


def build_candidate_maps(candidates: list[dict]) -> tuple[set[int], dict[str, int]]:
    ids = {int(item["candidate_id"]) for item in candidates}
    code_to_id = {
        str(item["target_code"]).zfill(6): int(item["candidate_id"])
        for item in candidates
    }
    return ids, code_to_id


def compact_from_rich_teacher_answer(
    answer: object,
    candidates: list[dict],
) -> list[list] | None:
    items = parse_json_safe(answer)
    if not isinstance(items, list):
        return None

    expected_ids, code_to_id = build_candidate_maps(candidates)
    compact = []
    for item in items:
        if not isinstance(item, dict):
            return None

        candidate_id = safe_int(item.get("candidate_id"), default=0)
        if candidate_id not in expected_ids:
            code = str(item.get("target_code", "")).strip().zfill(6)
            candidate_id = code_to_id.get(code, 0)
        if candidate_id not in expected_ids:
            return None

        action = normalize_action_to_code(
            item.get("topology_action", item.get("action", ""))
        )
        if action not in VALID_ACTION_CODES:
            return None

        direction = safe_int(item.get("impact_direction", 0))
        if direction not in VALID_DIRECTIONS:
            direction = 0

        strength = (
            clamp_bucket_0_3(item.get("strength_bucket"))
            if "strength_bucket" in item
            else continuous_to_bucket_0_3(item.get("impact_magnitude", 0.0))
        )
        confidence = (
            clamp_bucket_0_3(item.get("confidence_bucket"))
            if "confidence_bucket" in item
            else continuous_to_bucket_0_3(item.get("confidence", 0.0))
        )
        compact.append([candidate_id, action, direction, strength, confidence])

    output_ids = [item[0] for item in compact]
    if (
        len(output_ids) != len(expected_ids)
        or len(set(output_ids)) != len(output_ids)
        or set(output_ids) != expected_ids
    ):
        return None
    return sorted(compact, key=lambda item: item[0])


def compact_from_teacher_compact_answer(
    answer: object,
    candidates: list[dict],
) -> list[list] | None:
    items = parse_json_safe(answer)
    if not isinstance(items, list):
        return None

    expected_ids, _ = build_candidate_maps(candidates)
    compact = []
    for item in items:
        if not isinstance(item, list) or len(item) < 6:
            return None

        candidate_id = safe_int(item[0])
        action = normalize_action_to_code(item[1])
        direction = safe_int(item[2])
        if candidate_id not in expected_ids or action not in VALID_ACTION_CODES:
            return None
        if direction not in VALID_DIRECTIONS:
            direction = 0

        if len(item) == 6:
            strength = clamp_bucket_0_3(item[3])
            confidence = clamp_bucket_0_3(item[4])
        else:
            strength = continuous_to_bucket_0_3(item[3])
            confidence = continuous_to_bucket_0_3(item[4])

        compact.append([candidate_id, action, direction, strength, confidence])

    output_ids = [item[0] for item in compact]
    if (
        len(output_ids) != len(expected_ids)
        or len(set(output_ids)) != len(output_ids)
        or set(output_ids) != expected_ids
    ):
        return None
    return sorted(compact, key=lambda item: item[0])


def build_compact_output(row: pd.Series) -> tuple[list[list] | None, str]:
    candidates_json = None
    for column in (
        "Candidates_JSON_Student",
        "Candidates_JSON_Teacher",
        "Candidates_JSON",
    ):
        if column in row and isinstance(row[column], str):
            candidates_json = row[column]
            break

    candidates = parse_candidates(candidates_json)
    if not candidates:
        return None, "BAD_CANDIDATES"

    compact = compact_from_rich_teacher_answer(
        row.get("Teacher_Answer", ""),
        candidates,
    )
    if compact is not None:
        return compact, "FROM_RICH"

    compact = compact_from_teacher_compact_answer(
        row.get("Teacher_Compact_Answer", ""),
        candidates,
    )
    if compact is not None:
        return compact, "FROM_COMPACT"
    return None, "BAD_TEACHER_ANSWER"


def build_sft_data(dataset: str = DEFAULT_DATASET) -> list[Path]:
    root = dataset_dir(dataset)
    specs = [
        (
            "train",
            root / f"llm_exam_train_teacher_{dataset}_answered.csv",
            root / f"llm_exam_train_student_{dataset}.csv",
            root / f"sft_data_train_compact_{dataset}.jsonl",
            root / f"sft_data_train_compact_audit_{dataset}.csv",
        ),
        (
            "val",
            root / f"llm_exam_val_teacher_{dataset}_answered.csv",
            root / f"llm_exam_val_student_{dataset}.csv",
            root / f"sft_data_val_compact_{dataset}.jsonl",
            root / f"sft_data_val_compact_audit_{dataset}.csv",
        ),
    ]

    outputs = []
    summaries = []
    for split, teacher_path, student_path, output_path, audit_path in specs:
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher answer file not found: {teacher_path}")
        if not student_path.exists():
            raise FileNotFoundError(f"Student prompt file not found: {student_path}")

        teacher = pd.read_csv(teacher_path, dtype={"Symbol": str})
        student = pd.read_csv(student_path, dtype={"Symbol": str})
        teacher["Symbol"] = teacher["Symbol"].astype(str).str.zfill(6)
        student["Symbol"] = student["Symbol"].astype(str).str.zfill(6)

        required_teacher = [
            "Date",
            "Symbol",
            "Teacher_Answer",
            "Teacher_Compact_Answer",
            "Annotate_Status",
            "Candidates_JSON",
        ]
        required_student = ["Date", "Symbol", "Student_Prompt", "Candidates_JSON"]
        missing_teacher = [c for c in required_teacher if c not in teacher.columns]
        missing_student = [c for c in required_student if c not in student.columns]
        if missing_teacher:
            raise ValueError(f"{teacher_path} is missing columns: {missing_teacher}")
        if missing_student:
            raise ValueError(f"{student_path} is missing columns: {missing_student}")

        teacher = teacher[teacher["Annotate_Status"] == "SUCCESS"].copy()
        teacher = teacher.rename(
            columns={"Candidates_JSON": "Candidates_JSON_Teacher"}
        )
        student = student.rename(
            columns={"Candidates_JSON": "Candidates_JSON_Student"}
        )
        merged = pd.merge(
            student,
            teacher[
                [
                    "Date",
                    "Symbol",
                    "Teacher_Answer",
                    "Teacher_Compact_Answer",
                    "Annotate_Status",
                    "Candidates_JSON_Teacher",
                ]
            ],
            on=["Date", "Symbol"],
            how="inner",
        )

        entries = []
        audit_rows = []
        drop_reasons: Counter = Counter()
        for _, row in merged.iterrows():
            compact, source_type = build_compact_output(row)
            if compact is None:
                drop_reasons[source_type] += 1
                continue

            output = json.dumps(compact, ensure_ascii=False)
            entries.append(
                {
                    "instruction": str(row["Student_Prompt"]),
                    "input": "",
                    "output": output,
                }
            )
            audit_rows.append(
                {
                    "Date": row["Date"],
                    "Symbol": row["Symbol"],
                    "Output": output,
                    "Source_Type": source_type,
                    "Num_Candidates": len(compact),
                }
            )

        with output_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if SAVE_AUDIT:
            pd.DataFrame(audit_rows).to_csv(
                audit_path,
                index=False,
                encoding="utf_8_sig",
            )

        outputs.append(output_path)
        summaries.append(
            f"{split}: {len(entries)} samples, {sum(drop_reasons.values())} dropped"
        )

    print(f"SFT data saved for {dataset} ({'; '.join(summaries)})")
    return outputs


def main() -> None:
    args = parse_args()
    build_sft_data(args.dataset)


if __name__ == "__main__":
    main()
