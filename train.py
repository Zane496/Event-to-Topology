"""Downstream Full Model orchestration in the original E2T code layout."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.data import DualGraphDataset
from utils.metrics import evaluate
from utils.train_utils import train_one, aggregate_results
SEEDS = (3407, 3408, 3409, 3410, 3411)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_TAG = "brn_all"

def configure_dataset(dataset):
    """Use original project data/ hierarchy with standard *_all naming."""
    global DATASET_TAG, DATA_DIR, FEATURES_CSV, PRIOR_CSV, NODES_CSV, LURI_DOWNSTREAM_DIR, LURI_SIGNAL
    global OUTPUT_DIR, CKPT_DIR
    if dataset not in ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all"):
        raise ValueError(f"Unsupported full-event dataset: {dataset}")
    DATASET_TAG = dataset
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", dataset))
    FEATURES_CSV = os.path.join(DATA_DIR, f"{dataset}_features.csv")
    PRIOR_CSV = os.path.join(DATA_DIR, f"{dataset}_multi_relational_zero_shot_edges.csv")
    NODES_CSV = os.path.join(DATA_DIR, f"{dataset}_nodes.csv")
    LURI_DOWNSTREAM_DIR = os.path.join(DATA_DIR, f"downstream_inputs_{dataset}")
    LURI_SIGNAL = {"name": "E2T", "graph": os.path.join(LURI_DOWNSTREAM_DIR, f"dynamic_4rel_graphs_{dataset}.npy"),
                   "dates": os.path.join(LURI_DOWNSTREAM_DIR, f"graph_dates_{dataset}.csv")}
    OUTPUT_DIR = os.path.join(DATA_DIR, f"downstream_results_{dataset}")
    CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    return LURI_SIGNAL


configure_dataset(DATASET_TAG)
FULL_MODEL = {"name": "Full Model"}

def run(dataset="brn_all", seeds=SEEDS):
    signal = configure_dataset(dataset)
    required = [FEATURES_CSV, NODES_CSV, PRIOR_CSV, signal["graph"], signal["dates"]]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing inputs for Full Model. Generate all-announcement data and Step7 first:\n" + "\n".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(FULL_MODEL, signal=signal)
    print(f"E2T Full Model | {dataset} | seeds={tuple(seeds)} | device={device}")
    datasets = {split: DualGraphDataset(signal, split, dataset) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(datasets[split], batch_size=1, shuffle=False) for split in datasets}
    rows=[]
    for seed in seeds:
        model, best_epoch, best_score = train_one(cfg, loaders["train"], loaders["val"], datasets["train"].feature_dim, seed, device, OUTPUT_DIR, CKPT_DIR)
        test = evaluate(model, loaders["test"], device)
        rows.append({"Dataset": dataset, "Seed": seed, "Configuration": "Full Model", "Best_Epoch": best_epoch,
                     "Best_SelectionScore": best_score, **{k: test[k] for k in ("E_IC","E_SR","G_IC","G_SP","MDD","Calmar")}})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f"full_model_per_seed_partial_{dataset}.csv"), index=False, encoding="utf_8_sig")
        print(f"seed={seed} | E-IC={test['E_IC']:.4f} | G-IC={test['G_IC']:.4f} | G-SP={test['G_SP']:.4f} | Calmar={test['Calmar']:.4f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    per_seed = pd.DataFrame(rows)
    summary = aggregate_results(per_seed)
    per_seed.to_csv(os.path.join(OUTPUT_DIR, f"full_model_per_seed_{dataset}.csv"), index=False, encoding="utf_8_sig")
    summary.to_csv(os.path.join(OUTPUT_DIR, f"full_model_summary_{dataset}.csv"), index=False, encoding="utf_8_sig")
    print(summary.to_string(index=False))
    return per_seed, summary



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run E2T Full Model only")
    p.add_argument("--dataset", default="brn_all", choices=("cev_all","inu_all","emd_all","brn_all","mro_all"))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = p.parse_args()
    run(args.dataset, tuple(args.seeds))
