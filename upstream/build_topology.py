"""Convert Student predictions into E2T Full-Model adjusted adjacency tensors.

The parsing rules, action normalization, isolation handling, topology
attenuation, and output tensor layout are retained from Step7. Downstream
uses relation actions only; no event-feature tensors are injected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
VERBOSE = False
DATASET = DEFAULT_DATASET

CHANNEL_MAPPING = {
    "SUPPLY_CHAIN": 0,
    "COMPETITOR": 1,
    "PARTNERSHIP": 2,
    "CAPITAL": 3,
}
DECOUPLED_RETAINED_WEIGHT = 0.20
ISOLATED_RETAINED_WEIGHT = 0.00
FORCE_ISOLATED_TO_ZERO = True  # Retained from source parser; non-action fields are upstream-only.


def log(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E2T topology tensors.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def configure_paths(dataset: str) -> None:
    global DATASET, DATA_DIR, NODES_FILE_CANDIDATES
    global PRIOR_GRAPH_FILE_CANDIDATES, PRED_DIR, PRED_RESULT_FILES, OUTPUT_DIR

    DATASET = dataset
    DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / dataset)
    NODES_FILE_CANDIDATES = [
        os.path.join(DATA_DIR, f"{dataset}_nodes.csv"),
    ]
    PRIOR_GRAPH_FILE_CANDIDATES = [
        os.path.join(
            DATA_DIR,
            f"{dataset}_multi_relational_zero_shot_edges.csv",
        ),
    ]
    PRED_DIR = DATA_DIR
    PRED_RESULT_FILES = [
        {
            "period": "train",
            "paths": [
                os.path.join(
                    PRED_DIR,
                    f"test_results_raw_train_OOF_Bstyle_compact_{dataset}.csv",
                )
            ],
        },
        {
            "period": "val",
            "paths": [
                os.path.join(
                    PRED_DIR,
                    f"test_results_raw_val_B_compact_{dataset}.csv",
                )
            ],
        },
        {
            "period": "test",
            "paths": [
                os.path.join(
                    PRED_DIR,
                    f"test_results_raw_test_B_compact_{dataset}.csv",
                )
            ],
        },
    ]
    OUTPUT_DIR = os.path.join(DATA_DIR, f"downstream_inputs_{dataset}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


configure_paths(DEFAULT_DATASET)

def resolve_existing_path(candidates, required=True, name="file"):
    for path in candidates:
        if os.path.exists(path):
            return path

    if required:
        raise FileNotFoundError(f"Could not find {name}. Candidate paths: {candidates}")

    return None


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm_code(x):
    return str(x).strip().zfill(6)


def norm_date(x):
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def normalize_action_code(x):
    x = str(x).strip().upper()

    if x in {"P", "PRESERVED", "KEEP"}:
        return "P"

    if x in {"D", "DECOUPLED", "WEAKEN", "WEAKENED"}:
        return "D"

    if x in {"I", "ISOLATED", "NULL", "NONE"}:
        return "I"

    return "INVALID"


def action_full_name(action_code):
    mapping = {
        "P": "PRESERVED",
        "D": "DECOUPLED",
        "I": "ISOLATED",
    }
    return mapping.get(action_code, "UNKNOWN")


def clip_bucket_0_3(x):
    v = safe_int(x, default=0)
    return int(max(0, min(3, v)))


def bucket_to_value(bucket):
    bucket = int(max(0, min(3, bucket)))
    return bucket / 3.0


def extract_json_array_robust(text):
    try:
        text = str(text).strip()
        text = (
            text
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .replace("\ufeff", "")
            .strip()
        )

        left = text.find("[")

        if left == -1:
            return None

        candidate = text[left:].strip()


        right = candidate.rfind("]")

        if right != -1:
            normal_candidate = candidate[:right + 1]

            try:
                return json.loads(normal_candidate)
            except json.JSONDecodeError:
                pass

        if (
            candidate.startswith("[[")
            and candidate.endswith("]}")
            and candidate.count("[") == candidate.count("]") + 1
        ):
            repaired_candidate = candidate[:-1] + "]"

            try:
                return json.loads(repaired_candidate)
            except json.JSONDecodeError:
                return None

        return None

    except Exception:
        return None


def parse_candidates_json(x):
    """
    Candidates_JSON:
    [
      {
        "candidate_id": 1,
        "target_code": "002197",
        "target_name": "...",
        "relation_type": "COMPETITOR",
        "is_prior_neighbor": 1
      }
    ]
    """
    try:
        if not isinstance(x, str):
            x = str(x)

        candidates = json.loads(x)

        if not isinstance(candidates, list):
            return {}

        out = {}

        for item in candidates:
            if not isinstance(item, dict):
                continue

            cid = safe_int(item.get("candidate_id", 0))

            if cid <= 0:
                continue

            out[cid] = {
                "candidate_id": cid,
                "target_code": norm_code(item.get("target_code", "")),
                "target_name": str(item.get("target_name", "")).strip(),
                "relation_type": str(item.get("relation_type", "NULL")).strip(),
                "is_prior_neighbor": safe_int(item.get("is_prior_neighbor", 0)),
            }

        return out

    except Exception:
        return {}


def parse_compact_output(raw_output):
    """
    Parse compact Student outputs in the form:
        [[i, a, d, s, p], ...]

    The legacy six-field form is also accepted:
        [[i, a, d, s, p, r], ...]
    """
    parsed = extract_json_array_robust(raw_output)

    if not isinstance(parsed, list):
        return None

    results = []

    for item in parsed:
        if isinstance(item, list):
            if len(item) < 5:
                return None

            cid = safe_int(item[0])
            action = normalize_action_code(item[1])
            direction = safe_int(item[2])
            strength_bucket = clip_bucket_0_3(item[3])
            confidence_bucket = clip_bucket_0_3(item[4])

        elif isinstance(item, dict):
            cid = safe_int(item.get("i", item.get("candidate_id", 0)))
            action = normalize_action_code(
                item.get("a", item.get("action", item.get("topology_action", "")))
            )
            direction = safe_int(item.get("d", item.get("impact_direction", 0)))
            strength_bucket = clip_bucket_0_3(
                item.get(
                    "s", item.get("strength_bucket", item.get("impact_magnitude", 0))
                )
            )
            confidence_bucket = clip_bucket_0_3(
                item.get("p", item.get("confidence_bucket", item.get("confidence", 0)))
            )

        else:
            return None

        if cid <= 0:
            return None

        if action not in {"P", "D", "I"}:
            return None

        if direction not in {-1, 0, 1}:
            direction = 0

        if FORCE_ISOLATED_TO_ZERO and action == "I":
            direction = 0
            strength_bucket = 0
            confidence_bucket = 0

        results.append(
            {
                "candidate_id": cid,
                "action_code": action,
                "direction": direction,
                "strength_bucket": strength_bucket,
                "confidence_bucket": confidence_bucket,
            }
        )

    return results


def get_prompt_col(df):
    if "Student_Prompt" in df.columns:
        return "Student_Prompt"
    if "Prompt" in df.columns:
        return "Prompt"
    return None


def get_name_col(df):
    if "Name" in df.columns:
        return "Name"
    if "名称" in df.columns:
        return "名称"
    return None


def load_nodes():
    nodes_file = resolve_existing_path(
        NODES_FILE_CANDIDATES,
        required=True,
        name="dataset_nodes.csv",
    )

    df_nodes = pd.read_csv(nodes_file, dtype={"Symbol": str})

    if "Symbol" not in df_nodes.columns and "代码" in df_nodes.columns:
        df_nodes = df_nodes.rename(columns={"代码": "Symbol"})

    if "Symbol" not in df_nodes.columns:
        raise ValueError("The node file must contain either a Symbol or 代码 column.")

    df_nodes["Symbol"] = df_nodes["Symbol"].astype(str).str.zfill(6)
    df_nodes = df_nodes.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    ordered_codes = df_nodes["Symbol"].tolist()
    code_to_idx = {code: idx for idx, code in enumerate(ordered_codes)}

    node_index_path = os.path.join(OUTPUT_DIR, f"node_index_{DATASET}.csv")
    df_nodes[["Symbol"]].to_csv(node_index_path, index=False, encoding="utf_8_sig")

    log(f"Loaded {len(ordered_codes)} nodes.")
    log(f"Saved node index: {node_index_path}")

    return ordered_codes, code_to_idx


def load_static_graph(code_to_idx):
    prior_file = resolve_existing_path(
        PRIOR_GRAPH_FILE_CANDIDATES,
        required=True,
        name="multi_relational_zero_shot_edges.csv",
    )

    df_prior = pd.read_csv(
        prior_file,
        dtype={
            "Source": str,
            "Target": str,
        },
    )

    required_cols = ["Source", "Target", "Relation_Type"]
    missing = [c for c in required_cols if c not in df_prior.columns]

    if missing:
        raise ValueError(f"The prior-graph file is missing required columns: {missing}")

    num_nodes = len(code_to_idx)
    num_relations = len(CHANNEL_MAPPING)

    A_static = np.zeros(
        (num_relations, num_nodes, num_nodes),
        dtype=np.float32,
    )

    prior_edge_relation = {}

    valid_edges = 0

    for _, row in df_prior.iterrows():
        src = norm_code(row["Source"])
        tgt = norm_code(row["Target"])
        rel_type = str(row["Relation_Type"]).strip()

        if rel_type not in CHANNEL_MAPPING:
            continue

        if src not in code_to_idx or tgt not in code_to_idx:
            continue

        i = code_to_idx[src]
        j = code_to_idx[tgt]
        c = CHANNEL_MAPPING[rel_type]

        A_static[c, i, j] = 1.0
        A_static[c, j, i] = 1.0

        prior_edge_relation[(src, tgt)] = rel_type
        prior_edge_relation[(tgt, src)] = rel_type

        valid_edges += 1

    static_path = os.path.join(OUTPUT_DIR, f"static_4rel_graph_{DATASET}.npy")
    np.save(static_path, A_static)

    log(f"Loaded static prior graph with {valid_edges} valid edge records.")
    log(f"Static graph tensor shape: {A_static.shape}")
    log(f"Saved static graph: {static_path}")

    return A_static, prior_edge_relation

def resolve_pred_file(period_item):
    for path in period_item["paths"]:
        if os.path.exists(path):
            return path
    return None


def load_and_parse_all_predictions():
    all_event_rows = []
    all_edge_rows = []

    total_events = 0
    parsed_events = 0
    skipped_events = 0

    for period_item in PRED_RESULT_FILES:
        period = period_item["period"]
        file_path = resolve_pred_file(period_item)

        if file_path is None:
            raise FileNotFoundError(
                f"Could not find the {period} prediction file.\n"
                f"Candidate paths: {period_item['paths']}\n"
                "Topology construction requires the Train OOF, validation, and "
                "test prediction files. No split may be omitted or replaced by "
                "a legacy test file."
            )

        log(f"Reading {period} predictions: {file_path}")

        df = pd.read_csv(
            file_path,
            dtype={
                "Symbol": str,
            },
        )

        if df.empty:
            log(f"Prediction file is empty and will be skipped: {file_path}")
            continue

        required_cols = ["Date", "Symbol", "Raw_Output", "Candidates_JSON"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"{file_path} is missing required columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        name_col = get_name_col(df)

        for _, row in df.iterrows():
            total_events += 1

            try:
                date = norm_date(row["Date"])
                source_code = norm_code(row["Symbol"])
                source_name = str(row.get(name_col, "")) if name_col else ""

                candidates = parse_candidates_json(row["Candidates_JSON"])

                if not candidates:
                    skipped_events += 1

                    if skipped_events <= 20:
                        log("Failed to parse Candidates_JSON.")
                        log(f"   Period : {period}")
                        log(f"   Date   : {row.get('Date', '')}")
                        log(f"   Symbol : {row.get('Symbol', '')}")
                        log(
                            f"   Value  : "
                            f"{str(row.get('Candidates_JSON', ''))[:800]}"
                        )

                    continue

                pred_items = parse_compact_output(row["Raw_Output"])

                if pred_items is None:
                    skipped_events += 1

                    if skipped_events <= 20:
                        log("Failed to parse Raw_Output.")
                        log(f"   Period : {period}")
                        log(f"   Date   : {row.get('Date', '')}")
                        log(f"   Symbol : {row.get('Symbol', '')}")
                        log(
                            f"   Value  : "
                            f"{str(row.get('Raw_Output', ''))[:800]}"
                        )

                    continue

                pred_ids = [
                    item["candidate_id"]
                    for item in pred_items
                ]

                expected_ids = set(candidates.keys())

                is_complete = (
                        len(pred_ids) == len(expected_ids)
                        and len(set(pred_ids)) == len(pred_ids)
                        and set(pred_ids) == expected_ids
                )

                if not is_complete:
                    skipped_events += 1

                    if skipped_events <= 20:
                        log("Student output contains incomplete candidate IDs.")
                        log(f"   Period       : {period}")
                        log(f"   Date         : {date}")
                        log(f"   Symbol       : {source_code}")
                        log(f"   Expected IDs : {sorted(expected_ids)}")
                        log(f"   Predicted IDs: {pred_ids}")

                    continue

                parsed_events += 1

                all_event_rows.append(
                    {
                        "Date": date,
                        "Period": period,
                        "Symbol": source_code,
                        "Name": source_name,
                        "Raw_File": file_path,
                        "Raw_Output": row["Raw_Output"],
                        "Candidates_JSON": row["Candidates_JSON"],
                    }
                )

                seen_cids = set()

                for item in pred_items:
                    cid = item["candidate_id"]

                    if cid in seen_cids:
                        continue

                    seen_cids.add(cid)

                    if cid not in candidates:
                        continue

                    cand = candidates[cid]

                    target_code = cand["target_code"]
                    target_name = cand["target_name"]
                    rel_type = cand["relation_type"]

                    action = item["action_code"]
                    direction = item["direction"]
                    strength_bucket = item["strength_bucket"]
                    confidence_bucket = item["confidence_bucket"]

                    strength_value = bucket_to_value(strength_bucket)
                    confidence_value = bucket_to_value(confidence_bucket)

                    shock_score = direction * strength_value * confidence_value

                    all_edge_rows.append(
                        {
                            "Date": date,
                            "Period": period,
                            "Source_Code": source_code,
                            "Source_Name": source_name,
                            "Candidate_ID": cid,
                            "Target_Code": target_code,
                            "Target_Name": target_name,
                            "Relation_Type": rel_type,
                            "Is_Prior_Neighbor": cand["is_prior_neighbor"],
                            "Action_Code": action,
                            "Action": action_full_name(action),
                            "Direction": direction,
                            "Strength_Bucket": strength_bucket,
                            "Confidence_Bucket": confidence_bucket,
                            "Strength_Value": strength_value,
                            "Confidence_Value": confidence_value,
                            "Shock_Score": shock_score,
                            "Raw_File": file_path,
                        }
                    )

            except Exception as exc:

                skipped_events += 1

                if skipped_events <= 20:
                    log("Unexpected error while parsing an event.")

                    log(f"   Period : {period}")

                    log(f"   Date   : {row.get('Date', '')}")

                    log(f"   Symbol : {row.get('Symbol', '')}")

                    log(f"   Error  : {type(exc).__name__}: {exc}")

                    log(f"   Candidates_JSON: {str(row.get('Candidates_JSON', ''))[:500]}")

                    log(f"   Raw_Output     : {str(row.get('Raw_Output', ''))[:500]}")

                continue

    df_events = pd.DataFrame(all_event_rows)
    df_edges = pd.DataFrame(all_edge_rows)

    log("Student prediction parsing summary")
    log(f"   Total event rows : {total_events}")
    log(f"   Parsed events    : {parsed_events}")
    log(f"   Skipped events   : {skipped_events}")
    log(f"   Parsed edge rows : {len(df_edges)}")

    if df_edges.empty:
        raise RuntimeError(
            "No compact prediction edges were parsed. Check Raw_Output and "
            "Candidates_JSON."
        )

    edge_csv = os.path.join(OUTPUT_DIR, f"student_event_edges_{DATASET}.csv")
    df_edges.to_csv(edge_csv, index=False, encoding="utf_8_sig")

    event_csv = os.path.join(OUTPUT_DIR, f"student_event_rows_{DATASET}.csv")
    df_events.to_csv(event_csv, index=False, encoding="utf_8_sig")

    log(f"Saved target-level Student edges: {edge_csv}")
    log(f"Saved event-level Student rows: {event_csv}")

    log("Action distribution:")
    log(df_edges["Action"].value_counts(dropna=False).to_string())

    log("Direction distribution:")
    log(df_edges["Direction"].value_counts(dropna=False).to_string())

    log("Strength-bucket distribution:")
    log(df_edges["Strength_Bucket"].value_counts(dropna=False).to_string())

    log("Confidence-bucket distribution:")
    log(df_edges["Confidence_Bucket"].value_counts(dropna=False).to_string())

    log("Period distribution:")
    log(df_edges["Period"].value_counts(dropna=False).to_string())

    expected_periods = {"train", "val", "test"}
    actual_periods = set(df_edges["Period"].dropna().astype(str).unique())
    missing_periods = expected_periods - actual_periods
    if missing_periods:
        raise RuntimeError(
            f"Topology inputs are missing splits: {sorted(missing_periods)}. "
            "Confirm that generate_oof.py produced Train OOF, validation, and "
            "test prediction files."
        )

    return df_events, df_edges



def build_downstream_tensors(df_edges, code_to_idx, A_static, prior_edge_relation):
    """Reference BRN Full-Model attenuation; no event feature injection."""
    unique_dates = sorted(df_edges["Date"].dropna().astype(str).unique())
    if not unique_dates:
        raise RuntimeError("No event dates available for topology construction.")
    dynamic_graphs = np.empty((len(unique_dates), *A_static.shape), dtype=np.float32)
    changed_rows = []
    for t, date in enumerate(tqdm(unique_dates, desc="Building event-adjusted graphs")):
        adj = A_static.copy()
        rows = df_edges.loc[df_edges["Date"] == date]
        for _, row in rows.iterrows():
            src, tgt = norm_code(row["Source_Code"]), norm_code(row["Target_Code"])
            if src not in code_to_idx or tgt not in code_to_idx:
                continue
            action = str(row["Action_Code"]).strip().upper()
            if action not in {"D", "I"}:
                continue
            relation = prior_edge_relation.get((src, tgt))
            if relation is None:
                relation = str(row.get("Relation_Type", "NULL")).strip().upper()
            if relation not in CHANNEL_MAPPING:
                continue
            channel = CHANNEL_MAPPING[relation]
            i, j = code_to_idx[src], code_to_idx[tgt]
            if A_static[channel, i, j] <= 0:
                continue   # no new edges; non-edge candidate acts only in upstream.
            retained = DECOUPLED_RETAINED_WEIGHT if action == "D" else ISOLATED_RETAINED_WEIGHT
            weight = min(float(adj[channel, i, j]), float(retained))
            adj[channel, i, j] = adj[channel, j, i] = weight
            changed_rows.append({"Date": date, "Source_Code": src, "Target_Code": tgt,
                                 "Relation_Type": relation, "Action_Code": action,
                                 "Original_Weight": float(A_static[channel, i, j]),
                                 "Effective_Weight": weight})
        dynamic_graphs[t] = adj
    pd.DataFrame(changed_rows, columns=["Date", "Source_Code", "Target_Code", "Relation_Type",
                                               "Action_Code", "Original_Weight", "Effective_Weight"]).to_csv(
        os.path.join(OUTPUT_DIR, f"topology_damped_edges_{DATASET}.csv"), index=False, encoding="utf_8_sig")
    return unique_dates, dynamic_graphs



def build_topology(dataset: str = DEFAULT_DATASET) -> Path:
    configure_paths(dataset)
    _, code_to_idx = load_nodes()
    static_graph, prior_edge_relation = load_static_graph(code_to_idx)
    _, edges = load_and_parse_all_predictions()
    dates, dynamic_graphs = build_downstream_tensors(edges, code_to_idx, static_graph, prior_edge_relation)
    graph_path = os.path.join(OUTPUT_DIR, f"dynamic_4rel_graphs_{DATASET}.npy")
    graph_dates_path = os.path.join(OUTPUT_DIR, f"graph_dates_{DATASET}.csv")
    np.save(graph_path, dynamic_graphs)
    pd.DataFrame({"date": dates}).to_csv(graph_dates_path, index=False, encoding="utf_8_sig")
    print(f"[Step 7] Full Model only: {graph_path} shape={dynamic_graphs.shape}")
    return Path(OUTPUT_DIR)


def main() -> None:
    args = parse_args()
    build_topology(args.dataset)


if __name__ == "__main__":
    main()
