"""Construct candidate sets and Teacher/Student prompts for E2T."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
MAX_TOTAL_CANDIDATES = 8
MAX_PRIOR_NEIGHBORS = 6
MIN_NEGATIVES = 2
MAX_EMERGENT_NEGATIVES = 2
SHOW_PROGRESS = True

VALID_RELATIONS = {
    "SUPPLY_CHAIN",
    "COMPETITOR",
    "PARTNERSHIP",
    "CAPITAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E2T Teacher/Student prompts.")
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
        "nodes": data_dir / f"{dataset}_nodes.csv",
        "prior": data_dir / f"{dataset}_multi_relational_zero_shot_edges.csv",
        "train": data_dir / f"events_train_2022_2024_{dataset}.csv",
        "val": data_dir / f"events_val_2025H1_{dataset}.csv",
        "test": data_dir / f"events_test_2025H2_2026_{dataset}.csv",
        "train_teacher": data_dir / f"llm_exam_train_teacher_{dataset}.csv",
        "val_teacher": data_dir / f"llm_exam_val_teacher_{dataset}.csv",
        "test_teacher": data_dir / f"llm_exam_test_teacher_{dataset}.csv",
        "train_student": data_dir / f"llm_exam_train_student_{dataset}.csv",
        "val_student": data_dir / f"llm_exam_val_student_{dataset}.csv",
        "test_student": data_dir / f"llm_exam_test_student_{dataset}.csv",
    }


def normalize_relation_type(value: object) -> str | None:
    value = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "SUPPLYCHAIN": "SUPPLY_CHAIN",
        "SUPPLY_CHAIN": "SUPPLY_CHAIN",
        "COMPETITOR": "COMPETITOR",
        "COMPETITION": "COMPETITOR",
        "PARTNER": "PARTNERSHIP",
        "PARTNERSHIP": "PARTNERSHIP",
        "CAPITAL": "CAPITAL",
        "EQUITY": "CAPITAL",
        "SHAREHOLDING": "CAPITAL",
    }
    return aliases.get(value)


def stable_rng(source: str, date: str) -> random.Random:
    key = f"{source}_{date}"
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def clean_entity_name(name: object) -> str:
    suffixes = [
        "集团股份有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团有限公司",
        "有限公司",
        "股份",
        "集团",
        "控股",
        "公司",
        "金融科技",
        "数字科技",
        "网络科技",
        "信息科技",
        "信息技术",
        "信息服务",
        "软件服务",
        "科技",
        "信息",
        "软件",
        "网络",
        "数据",
        "数字",
        "智能",
        "城乡建设",
        "城市建设",
        "基础设施",
        "市政工程",
        "工程建设",
        "规划设计",
        "建筑设计",
        "建设工程",
        "城建",
        "建设",
        "规划",
        "设计",
        "工程",
        "建筑",
        "地产",
        "置业",
        "园林",
        "交通",
        "基建",
    ]
    result = str(name)
    for suffix in suffixes:
        result = result.replace(suffix, "")
    return result.strip()


def generate_teacher_prompt(row: pd.Series) -> str:
    source = str(row["Symbol"]).zfill(6)
    return f"""【冲击事件观测】
    发布主体: {row['Name']} ({source})
    发布日期: {row['Date']}

    【事前公开信息】
    {row['Student_Input']}

    【教师特权事件标签】
    Z={float(row['Z_Score']):.3f}
    Ret={float(row['Impact_Return']):.3f}
    ReactionDate={row['Impact_Reaction_Date']}
    Dir={int(row['Impact_Direction'])}
    Mag={float(row['Impact_Magnitude']):.3f}
    Conf={float(row['Impact_Confidence']):.3f}
    H={int(row['Impact_Horizon'])}
    S_Bucket={int(row['Impact_Strength_Bucket'])}
    P_Bucket={int(row['Impact_Confidence_Bucket'])}

    【候选节点】
    {row['Candidates_Teacher_Str']}

    【任务】
    对每个候选节点判断该事件对 target 的影响。

    只输出紧凑 JSON 数组：
    [[i,a,d,s,p,r],...]

    字段含义：
    i=候选编号；
    a=P维持/D弱化/I无关；
    d=-1负向/0中性/1正向；
    s=强度桶0-3；
    p=置信桶0-3；
    r=30字以内原因。

    规则：
    1. 必须覆盖所有候选编号，不得遗漏、不得新增。
    2. REL=NULL 且无业务传导路径时，通常 a=I,d=0,s=0。
    3. Dir 是 source 事件方向，target 方向需结合 REL 推断。
    4. reason 必须基于基本面和关系传导逻辑，不得直接复述 Z 或 Ret。
    5. 直接以 [ 开头，以 ] 结尾，不输出其他文字。"""


def generate_student_prompt(row: pd.Series) -> str:
    source = str(row["Symbol"]).zfill(6)
    return f"""事件:
主体={row['Name']}({source})
日期={row['Date']}
摘要:
{row['Student_Input']}

候选:
{row['Candidates_Student_Str']}

输出[[i,a,d,s,p]]
a=P维持/D弱化/I无关; d=-1负向/0中性/1正向; s=强度0-3; p=置信0-3。
只输出JSON数组，不写解释。"""


def build_prompts(dataset: str = DEFAULT_DATASET) -> list[tuple[Path, Path]]:
    paths = resolve_paths(dataset)
    for key in ("nodes", "prior"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Required file not found: {paths[key]}")

    nodes = pd.read_csv(paths["nodes"], dtype={"Symbol": str})
    nodes["Symbol"] = nodes["Symbol"].astype(str).str.zfill(6)
    nodes = (
        nodes[["Symbol", "Name"]]
        .dropna(subset=["Symbol", "Name"])
        .drop_duplicates(subset=["Symbol"])
        .sort_values("Symbol")
        .reset_index(drop=True)
    )
    node_dict = dict(zip(nodes["Symbol"], nodes["Name"]))
    all_codes = sorted(node_dict)

    prior = pd.read_csv(
        paths["prior"],
        dtype={"Source": str, "Target": str},
    )
    prior["Source"] = prior["Source"].astype(str).str.zfill(6)
    prior["Target"] = prior["Target"].astype(str).str.zfill(6)
    if "Relation_Type" not in prior.columns:
        raise ValueError("Prior graph is missing Relation_Type.")
    prior["Relation_Type"] = prior["Relation_Type"].apply(normalize_relation_type)
    prior = prior[prior["Relation_Type"].isin(VALID_RELATIONS)].copy()

    prior_adj: dict[str, dict[str, tuple[str, str]]] = {}
    for _, edge in prior.iterrows():
        source = edge["Source"]
        target = edge["Target"]
        relation = edge["Relation_Type"]
        if source not in node_dict or target not in node_dict:
            continue
        prior_adj.setdefault(source, {})[target] = (node_dict[target], relation)
        prior_adj.setdefault(target, {})[source] = (node_dict[source], relation)

    def build_candidates(source: str, student_text: str, date: str) -> list[dict]:
        rng = stable_rng(source, date)
        neighbors = [
            (code, value[0], value[1])
            for code, value in prior_adj.get(source, {}).items()
            if code in node_dict and code != source
        ]
        neighbors = sorted(neighbors, key=lambda item: item[0])
        if len(neighbors) > MAX_PRIOR_NEIGHBORS:
            neighbors = sorted(
                rng.sample(neighbors, MAX_PRIOR_NEIGHBORS),
                key=lambda item: item[0],
            )
        known_targets = {item[0] for item in neighbors}

        emergent = []
        for code in all_codes:
            if code == source or code in known_targets:
                continue
            name = node_dict[code]
            clean_name = clean_entity_name(name)
            if name in student_text or (len(clean_name) > 1 and clean_name in student_text):
                emergent.append(code)
        emergent = sorted(set(emergent))
        rng.shuffle(emergent)
        emergent = emergent[:MAX_EMERGENT_NEGATIVES]

        candidates = []
        for code, name, relation in neighbors:
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "target_code": code,
                    "target_name": name,
                    "relation_type": relation,
                    "is_prior_neighbor": 1,
                }
            )

        remaining_slots = MAX_TOTAL_CANDIDATES - len(candidates)
        for code in emergent:
            if remaining_slots <= 0:
                break
            candidates.append(
                {
                    "candidate_id": len(candidates) + 1,
                    "target_code": code,
                    "target_name": node_dict[code],
                    "relation_type": "NULL",
                    "is_prior_neighbor": 0,
                }
            )
            remaining_slots -= 1

        negative_count = sum(
            item["is_prior_neighbor"] == 0 for item in candidates
        )
        needed_random = max(0, MIN_NEGATIVES - negative_count)
        needed_random = min(
            needed_random,
            MAX_TOTAL_CANDIDATES - len(candidates),
        )
        used_codes = {item["target_code"] for item in candidates}
        non_neighbors = [
            code
            for code in all_codes
            if code != source
            and code not in used_codes
            and code not in known_targets
        ]
        if needed_random > 0 and non_neighbors:
            for code in rng.sample(
                non_neighbors,
                min(needed_random, len(non_neighbors)),
            ):
                if len(candidates) >= MAX_TOTAL_CANDIDATES:
                    break
                candidates.append(
                    {
                        "candidate_id": len(candidates) + 1,
                        "target_code": code,
                        "target_name": node_dict[code],
                        "relation_type": "NULL",
                        "is_prior_neighbor": 0,
                    }
                )
        return candidates

    def teacher_candidates(candidates: list[dict]) -> str:
        return "\n".join(
            f"{item['candidate_id']}. "
            f"[{item['target_code']} {item['target_name']}] "
            f"REL={item['relation_type']}"
            for item in candidates
        )

    def student_candidates(candidates: list[dict]) -> str:
        return "\n".join(
            f"{item['candidate_id']}|{item['target_code']}|"
            f"{item['target_name']}|{item['relation_type']}"
            for item in candidates
        )

    required_columns = [
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
    teacher_columns = [
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
        "Candidates_JSON",
        "Teacher_Prompt",
    ]
    student_columns = [
        "Date",
        "Symbol",
        "Name",
        "Candidates_JSON",
        "Student_Prompt",
    ]

    outputs = []
    split_specs = [
        ("train", paths["train"], paths["train_teacher"], paths["train_student"]),
        ("val", paths["val"], paths["val_teacher"], paths["val_student"]),
        ("test", paths["test"], paths["test_teacher"], paths["test_student"]),
    ]

    for split, input_file, teacher_file, student_file in split_specs:
        if not input_file.exists():
            raise FileNotFoundError(f"Event split not found: {input_file}")
        frame = pd.read_csv(input_file, dtype={"Symbol": str})
        frame["Symbol"] = frame["Symbol"].astype(str).str.zfill(6)
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{input_file} is missing columns: {missing}")

        candidates = []
        for _, row in tqdm(
            frame.iterrows(),
            total=len(frame),
            desc=f"Prompts {split}",
            disable=not SHOW_PROGRESS,
        ):
            candidates.append(
                build_candidates(
                    source=str(row["Symbol"]).zfill(6),
                    student_text=str(row["Student_Input"]),
                    date=str(row["Date"]),
                )
            )

        frame["Candidates_List"] = candidates
        frame["Candidates_JSON"] = frame["Candidates_List"].apply(
            lambda items: json.dumps(items, ensure_ascii=False)
        )
        frame["Candidates_Teacher_Str"] = frame["Candidates_List"].apply(
            teacher_candidates
        )
        frame["Candidates_Student_Str"] = frame["Candidates_List"].apply(
            student_candidates
        )
        frame["Teacher_Prompt"] = frame.apply(generate_teacher_prompt, axis=1)
        frame["Student_Prompt"] = frame.apply(generate_student_prompt, axis=1)

        frame[teacher_columns].to_csv(
            teacher_file,
            index=False,
            encoding="utf_8_sig",
        )
        frame[student_columns].to_csv(
            student_file,
            index=False,
            encoding="utf_8_sig",
        )
        outputs.append((teacher_file, student_file))

    print(f"Saved Teacher/Student prompts for {dataset} to {paths['nodes'].parent}")
    return outputs


def main() -> None:
    random.seed(42)
    args = parse_args()
    build_prompts(args.dataset)


if __name__ == "__main__":
    main()
