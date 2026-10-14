"""Market sequences, chronological splits, and aligned static/dynamic graph tensors."""
import os
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
IMPACT_WINDOW = (2, 3, 4, 5, 6)

class DualGraphDataset(Dataset):
    def __init__(self, signal, mode, dataset, lookback_window=10):
        super().__init__()
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset))
        NODES_CSV = os.path.join(dataset_dir, f"{dataset}_nodes.csv")
        FEATURES_CSV = os.path.join(dataset_dir, f"{dataset}_features.csv")
        PRIOR_CSV = os.path.join(dataset_dir, f"{dataset}_multi_relational_zero_shot_edges.csv")
        LURI_DOWNSTREAM_DIR = os.path.join(dataset_dir, f"downstream_inputs_{dataset}")
        DATASET_TAG = dataset
        self.signal_name = signal["name"]
        self.impact_window = set(IMPACT_WINDOW)

        nodes = pd.read_csv(NODES_CSV, dtype={"Symbol": str})
        if "Symbol" not in nodes.columns and "代码" in nodes.columns:
            nodes = nodes.rename(columns={"代码": "Symbol"})
        if "Symbol" not in nodes.columns:
            raise ValueError("nodes file must contain Symbol or 代码.")

        nodes["Symbol"] = nodes["Symbol"].astype(str).str.zfill(6)
        self.codes = nodes.drop_duplicates("Symbol")["Symbol"].tolist()
        # Step7 writes the exact node order used to build its adjacency tensors.
        # A dimension-only check cannot detect a reordered node list.
        index_file = os.path.join(LURI_DOWNSTREAM_DIR, f"node_index_{DATASET_TAG}.csv")
        if os.path.isfile(index_file):
            saved_codes = pd.read_csv(index_file, dtype={"Symbol": str})["Symbol"].astype(str).str.zfill(6).tolist()
            if saved_codes != self.codes:
                raise ValueError(
                    f"Node order mismatch between {NODES_CSV} and {index_file}. "
                    "Rebuild Step7 or restore the original node order."
                )
        code_to_idx = {code: idx for idx, code in enumerate(self.codes)}
        n = len(self.codes)

        df = pd.read_csv(FEATURES_CSV, dtype={"Symbol": str})
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(6)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(["Date", "Symbol"], keep="last")

        global_dates = sorted(df["Date"].drop_duplicates().tolist())
        date_to_idx = {date: idx for idx, date in enumerate(global_dates)}
        mux = pd.MultiIndex.from_product([global_dates, self.codes], names=["Date", "Symbol"])
        df = df.set_index(["Date", "Symbol"]).reindex(mux).reset_index().fillna(0)

        train_start = pd.Timestamp("2022-01-01")
        train_end = pd.Timestamp("2024-12-31")
        val_end = pd.Timestamp("2025-06-30")

        if mode == "train":
            target_dates = [date for date in global_dates if train_start <= date <= train_end]
        elif mode == "val":
            target_dates = [date for date in global_dates if train_end < date <= val_end]
        elif mode == "test":
            target_dates = [date for date in global_dates if date > val_end]
        else:
            raise ValueError("mode must be train / val / test.")

        df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        grouped = df.groupby("Symbol")

        df["Rank_Return"] = (
            grouped["Pct_Chg"].shift(-1).fillna(0.0)
            + grouped["Pct_Chg"].shift(-2).fillna(0.0)
            + grouped["Pct_Chg"].shift(-3).fillna(0.0)
        ) / 300.0
        df["PnL_Return_T1"] = grouped["Pct_Chg"].shift(-1).fillna(0.0) / 100.0
        df["Trade_Mask"] = pd.to_numeric(df["Is_Listed"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

        prev_close = grouped["Close"].shift(1)
        prev_vol = grouped["Volume"].shift(1)
        prev_amt = grouped["Amount"].shift(1)
        hl_range = df["High"] - df["Low"] + 1e-8

        df["Open_Gap"] = (df["Open"] - prev_close) / (prev_close + 1e-8)
        df["Close_Pos"] = (df["Close"] - df["Low"]) / hl_range
        df["Body_Pos"] = (df["Close"] - df["Open"]) / hl_range
        df["Volume_Ratio"] = np.log1p(df["Volume"] / (prev_vol + 1e-8))
        df["Amount_Ratio"] = np.log1p(df["Amount"] / (prev_amt + 1e-8))

        norm_cols = ["Open_Gap", "Close_Pos", "Body_Pos", "Volume_Ratio", "Amount_Ratio", "Amplitude", "Pct_Chg"]
        df.loc[df["Is_Listed"] != 1, norm_cols] = np.nan

        for col in norm_cols:
            mean = df.groupby("Date")[col].transform("mean")
            std = df.groupby("Date")[col].transform("std")
            df[col] = (df[col] - mean) / (std + 1e-8)

        df = df.fillna(0)

        feature_cols = ["Open_Gap", "Close_Pos", "Body_Pos", "Volume_Ratio", "Amount_Ratio", "Amplitude", "Pct_Chg", "Is_Listed", "Trade_Mask"]
        self.feature_dim = len(feature_cols) - 1

        df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
        features = df[feature_cols].values.reshape(len(global_dates), n, len(feature_cols)).astype(np.float32)
        rank_y = df["Rank_Return"].values.reshape(len(global_dates), n).astype(np.float32)
        pnl_y = df["PnL_Return_T1"].values.reshape(len(global_dates), n).astype(np.float32)

        channel = {"SUPPLY_CHAIN": 0, "COMPETITOR": 1, "PARTNERSHIP": 2, "CAPITAL": 3}
        static_adj = np.zeros((4, n, n), dtype=np.float32)

        prior = pd.read_csv(PRIOR_CSV, dtype={"Source": str, "Target": str})
        for _, row in prior.iterrows():
            src = str(row["Source"]).zfill(6)
            tgt = str(row["Target"]).zfill(6)
            rel = str(row["Relation_Type"]).strip().upper()
            if src in code_to_idx and tgt in code_to_idx and rel in channel:
                c, i, j = channel[rel], code_to_idx[src], code_to_idx[tgt]
                static_adj[c, i, j] = 1.0
                static_adj[c, j, i] = 1.0

        graph_arr = np.clip(np.load(signal["graph"]).astype(np.float32), 0.0, 1.0)
        if graph_arr.ndim != 4 or graph_arr.shape[1:] != (4, n, n):
            raise ValueError(f"{signal['name']} graph shape mismatch: {graph_arr.shape}")

        graph_dates = pd.to_datetime(pd.read_csv(signal["dates"])["date"]).tolist()
        if len(graph_dates) != graph_arr.shape[0]:
            raise ValueError(f"{signal['name']} graph_dates length mismatch.")

        self.samples = []

        for eval_date in target_dates:
            g_idx = date_to_idx[eval_date]
            if g_idx < lookback_window:
                continue

            active = []
            for event_idx, ann_date in enumerate(graph_dates):
                event_day_idx = bisect.bisect_left(global_dates, ann_date)
                lag = g_idx - event_day_idx
                if ann_date < eval_date and lag in self.impact_window:
                    active.append(event_idx)

            is_event = int(bool(active))
            dynamic_adj = np.min(graph_arr[active], axis=0).astype(np.float32) if active else static_adj.copy()
            x_window = np.transpose(features[g_idx - lookback_window + 1:g_idx + 1], (1, 0, 2))

            self.samples.append(
                (
                    torch.tensor(x_window[:, :, :-1], dtype=torch.float32),
                    torch.tensor(static_adj, dtype=torch.float32),
                    torch.tensor(dynamic_adj, dtype=torch.float32),
                    torch.tensor(rank_y[g_idx], dtype=torch.float32),
                    torch.tensor(pnl_y[g_idx], dtype=torch.float32),
                    torch.tensor(is_event, dtype=torch.long),
                    torch.tensor(x_window[:, -1, -1], dtype=torch.float32),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

