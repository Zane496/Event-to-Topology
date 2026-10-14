"""Build the four-relation static prior graph used by E2T.

PyCharm can run this file directly. Change ``DEFAULT_DATASET`` below when
needed, or override it from the command line with ``--dataset``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dashscope
import pandas as pd
from dashscope import Generation
from tqdm import tqdm

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
MODEL_NAME = "qwen-plus"
MAX_WORKERS = 5
MAX_RETRIES = 3
SHOW_PROGRESS = True

SYSTEM_PROMPT = """你是一个极其严谨的 A 股量化金融分析师，拥有深厚的产业链知识。
⚠️ 时间边界铁律：你只能基于 2022 年 1 月 1 日之前已公开的产业链上下游、同业竞争、战略合作、资本关联信息进行判断，绝对不得使用 2022 年及之后发生的任何合作、并购、业务转型、财报等事件。
"""

USER_PROMPT_TEMPLATE = """
我们正在构建多维金融知识图谱的先验基准。
【目标公司】：[{target_code} {target_name}]

【候选公司名单】：
{candidates}

【你的任务】：
请在候选名单中，挑选出与目标公司存在实质性关联的公司（最多 5 家）。
对于挑选出的每一家公司，你必须根据以下逻辑，赋予其 4 种【核心关系类型】之一。

⚠️ 极其重要：在 JSON 的 type 字段中，必须且只能输出【英文大写代码】，绝对不允许输出任何中文！

1. 产业链上下游（包括供应商、客户） -> 输出：SUPPLY_CHAIN
2. 同业竞争（核心业务存在直接竞争） -> 输出：COMPETITOR
3. 战略合作（非上下游的业务合作、技术研发等） -> 输出：PARTNERSHIP
4. 资本关联（母子公司、交叉持股等） -> 输出：CAPITAL

要求：
1. 宁缺毋滥，如果没有强关联，可以为空。
2. 必须严格以 JSON 格式输出，不要有任何其他解释文字。

【输出格式示例】：
{{
    "edges": [
        {{"code": "300308", "type": "COMPETITOR"}},
        {{"code": "601138", "type": "SUPPLY_CHAIN"}}
    ]
}}
"""

OUTPUT_COLUMNS = [
    "Source",
    "Source_Name",
    "Target",
    "Target_Name",
    "Weight",
    "Relation_Type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the E2T static prior graph.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def resolve_paths(dataset: str) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / dataset
    return (
        dataset_dir / f"{dataset}_nodes.csv",
        dataset_dir / f"{dataset}_multi_relational_zero_shot_edges.csv",
    )


def extract_json_from_text(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text.replace("\n", ""), re.IGNORECASE)
        return json.loads(match.group(0) if match else text)
    except Exception:
        return {"edges": []}


def ask_qwen_for_edges(
    target_code: str,
    target_name: str,
    candidates: str,
) -> tuple[str, str, list]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_code=target_code,
        target_name=target_name,
        candidates=candidates,
    )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = Generation.call(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                result_format="message",
                temperature=0.01,
                top_p=0.01,
                seed=42,
            )
            if response.status_code == 200:
                content = response.output.choices[0]["message"]["content"]
                payload = extract_json_from_text(content)
                return target_code, target_name, payload.get("edges", [])
            last_error = RuntimeError(
                f"API status {response.status_code}: {response.message}"
            )
        except Exception as exc:
            last_error = exc
        time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Prior-graph request failed for {target_code} after {MAX_RETRIES} attempts."
    ) from last_error


def build_prior_graph(dataset: str = DEFAULT_DATASET) -> Path:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set DASHSCOPE_API_KEY or QWEN_API_KEY before building the prior graph."
        )
    dashscope.api_key = api_key

    input_file, output_file = resolve_paths(dataset)
    if not input_file.exists():
        raise FileNotFoundError(f"Node file not found: {input_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_nodes = pd.read_csv(input_file, dtype={"Symbol": str})
    required = {"Symbol", "Name"}
    missing = required.difference(df_nodes.columns)
    if missing:
        raise ValueError(f"Node file is missing columns: {sorted(missing)}")

    df_nodes["Symbol"] = df_nodes["Symbol"].astype(str).str.zfill(6)
    node_dict = dict(zip(df_nodes["Symbol"], df_nodes["Name"]))
    candidates = ", ".join(
        f"[{code} {name}]" for code, name in node_dict.items()
    )

    processed_nodes: set[str] = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        existing = pd.read_csv(output_file, dtype={"Source": str})
        if "Source" in existing.columns:
            processed_nodes = set(existing["Source"].astype(str).str.zfill(6))
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            output_file,
            index=False,
            encoding="utf_8_sig",
        )

    remaining = {
        code: name for code, name in node_dict.items() if code not in processed_nodes
    }

    if remaining:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    ask_qwen_for_edges,
                    code,
                    name,
                    candidates,
                ): code
                for code, name in remaining.items()
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Prior graph",
                disable=not SHOW_PROGRESS,
            ):
                source, source_name, edges = future.result()
                rows = []
                for item in edges:
                    if isinstance(item, dict):
                        target = str(item.get("code", "")).strip()[:6].zfill(6)
                        relation = item.get("type", "Unknown")
                    else:
                        target = str(item).strip()[:6].zfill(6)
                        relation = "Unknown"

                    if target in node_dict and target != source:
                        rows.append(
                            {
                                "Source": source,
                                "Source_Name": source_name,
                                "Target": target,
                                "Target_Name": node_dict[target],
                                "Weight": 1.0,
                                "Relation_Type": relation,
                            }
                        )

                if rows:
                    pd.DataFrame(rows).to_csv(
                        output_file,
                        mode="a",
                        header=False,
                        index=False,
                        encoding="utf_8_sig",
                    )
                time.sleep(0.1)

    final_df = pd.read_csv(
        output_file,
        dtype={"Source": str, "Target": str},
    )
    if not final_df.empty:
        final_df["Source"] = final_df["Source"].astype(str).str.zfill(6)
        final_df["Target"] = final_df["Target"].astype(str).str.zfill(6)
        final_df = (
            final_df.drop_duplicates(subset=["Source", "Target"])
            .sort_values(["Source", "Target"])
            .reset_index(drop=True)
        )
    final_df.to_csv(output_file, index=False, encoding="utf_8_sig")
    print(f"Saved {len(final_df)} prior edges to {output_file}")
    return output_file


def main() -> None:
    args = parse_args()
    build_prior_graph(args.dataset)


if __name__ == "__main__":
    main()
