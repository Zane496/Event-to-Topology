"""Full Model training, validation checkpoint selection and seed aggregation."""
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from models.e2t import E2TFullModel
from utils.losses import CrossSectionalPairwiseRankingLoss
from utils.metrics import evaluate
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DYNAMIC_EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-4
PAIRWISE_ALPHA = 0.70

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_tag(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_").lower()

def train_one(cfg, train_loader, val_loader, feature_dim, seed, device, output_dir, ckpt_dir):
    set_seed(seed)

    model = E2TFullModel(feature_dim=feature_dim).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = CrossSectionalPairwiseRankingLoss(alpha=PAIRWISE_ALPHA)

    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    patience_count = 0
    history = []

    print("\n" + "=" * 100)
    print(f"Training           : {cfg['name']}")
    print(f"Seed               : {seed}")
    print(f"Signal             : {cfg['signal']['name']}")
    print("Checkpoint selector: event-only")
    print(f"Loss               : 0.5*H3 + 0.5*T1; each {PAIRWISE_ALPHA:.2f} Pair + {1.0 - PAIRWISE_ALPHA:.2f} MSE")
    print("=" * 100)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        sum_loss = sum_h3 = sum_t1 = 0.0
        steps = 0

        for x, static_adj, dynamic_adj, y_rank, y_pnl, _, trade_mask in train_loader:
            x = x.to(device)
            static_adj = static_adj.to(device)
            dynamic_adj = dynamic_adj.to(device)
            y_rank = y_rank.to(device)
            y_pnl = y_pnl.to(device)
            trade_mask = trade_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            score = model(x, static_adj, dynamic_adj)

            loss_h3 = criterion(score, y_rank, trade_mask)
            loss_t1 = criterion(score, y_pnl, trade_mask)
            loss = 0.5 * loss_h3 + 0.5 * loss_t1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            sum_loss += float(loss.item())
            sum_h3 += float(loss_h3.item())
            sum_t1 += float(loss_t1.item())
            steps += 1

        if steps == 0:
            raise RuntimeError(f"{cfg['name']} received no training samples.")

        val = evaluate(model, val_loader, device)
        selection_score = val["EventVal"]

        avg_loss = sum_loss / steps
        avg_h3 = sum_h3 / steps
        avg_t1 = sum_t1 / steps

        print(
            f"[{cfg['name']} | {epoch:03d}/{MAX_EPOCHS}] Loss={avg_loss:.5f} | "
            f"E-IC={val['E_IC']:.4f} | E-SR={val['E_SR']:.4f} | G-IC={val['G_IC']:.4f} | "
            f"G-SP={val['G_SP']:.4f} | MDD={val['MDD']*100:.2f}% | "
            f"Calmar={val['Calmar']:.4f} | "
            f"EventVal={val['EventVal']:.6f} | SELECT={selection_score:.6f}"
        )

        history.append(
            {
                "Epoch": epoch,
                "Loss": avg_loss,
                "Loss_H3": avg_h3,
                "Loss_T1": avg_t1,
                "G_IC_H3": val["G_IC"],
                "E_IC_H3": val["E_IC"],
                "G_IC_T1": val["G_T1"],
                "E_IC_T1": val["E_T1"],
                "G_SP": val["G_SP"],
                "E_SR": val["E_SR"],
                "MDD": val["MDD"],
                "Calmar": val["Calmar"],
                "Event_ValScore": val["EventVal"],
                "SelectionScore": selection_score,
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        min_epoch = MIN_DYNAMIC_EPOCHS
        if epoch >= min_epoch and patience_count >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {cfg['name']}.")

    model.load_state_dict(best_state)

    tag = safe_tag(cfg["name"])
    torch.save(
        {
            "state_dict": best_state,
            "seed": seed,
            "configuration": cfg["name"],
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "signal": cfg["signal"]["name"],
                                    "pairwise_alpha": PAIRWISE_ALPHA,
        },
        os.path.join(ckpt_dir, f"{tag}_seed_{seed}.pt"),
    )

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, f"{tag}_training_seed_{seed}.csv"),
        index=False,
        encoding="utf_8_sig",
    )

    return model, best_epoch, best_score

def aggregate_results(per_seed):
    rows = []
    for name in per_seed["Configuration"].drop_duplicates():
        df = per_seed[per_seed["Configuration"] == name]
        row = {"Configuration": name, "N_Seeds": len(df)}
        for metric in ("E_IC", "E_SR", "G_IC", "G_SP", "MDD", "Calmar"):
            values = df[metric].astype(float).to_numpy()
            row[f"{metric}_Mean"] = float(values.mean())
            row[f"{metric}_Std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)

