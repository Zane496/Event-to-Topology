"""Cross-sectional IC and friction-adjusted event/full-period portfolio metrics.

Turnover is an internal transaction-cost factor, not a reported metric.
"""
import numpy as np
import torch
from scipy.stats import spearmanr
TOP_K = 20

def safe_spearman_ic(pred, target, mask):
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    mask = np.asarray(mask).reshape(-1)

    valid = mask > 0
    if int(valid.sum()) <= 2:
        return None

    pred = pred[valid]
    target = target[valid]

    if np.std(pred) <= 1e-12 or np.std(target) <= 1e-12:
        return None

    ic = spearmanr(pred, target).statistic
    return float(ic) if np.isfinite(ic) else None

def portfolio_metrics(predictions, returns, masks, annualized_sharpe=True):
    daily_returns = []
    prev_long = set()
    total_friction = 0.0005 + 0.001

    for pred, true, mask in zip(predictions, returns, masks):
        pred = np.asarray(pred).reshape(-1)
        true = np.asarray(true).reshape(-1)
        mask = np.asarray(mask).reshape(-1)

        valid = np.where(mask > 0)[0]
        ranked = valid[np.argsort(pred[valid])[::-1]] if len(valid) else np.array([], dtype=int)
        current = set(ranked[:min(TOP_K, len(ranked))])

        long_ret = float(np.mean([true[idx] for idx in current])) if current else 0.0
        turnover = len(current - prev_long) / TOP_K if prev_long else (1.0 if current else 0.0)
        daily_returns.append(long_ret - turnover * total_friction)
        prev_long = current

    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) == 0:
        return {"Sharpe": 0.0, "MDD": 0.0, "Calmar": 0.0}

    mean_ret = float(np.mean(daily_returns))
    std_ret = float(np.std(daily_returns))
    sharpe_scale = np.sqrt(242) if annualized_sharpe else 1.0
    sharpe = mean_ret * sharpe_scale / std_ret if std_ret > 1e-12 else 0.0

    wealth = np.cumprod(1.0 + daily_returns)
    running_peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    drawdown = wealth / running_peak - 1.0
    mdd = float(min(0.0, np.min(drawdown))) if len(drawdown) else 0.0

    annual_return = mean_ret * 242.0
    calmar = annual_return / abs(mdd) if abs(mdd) > 1e-12 else 0.0
    return {"Sharpe": sharpe, "MDD": mdd, "Calmar": calmar}

def evaluate(model, loader, device):
    model.eval()

    g_h3, e_h3, g_t1, e_t1 = [], [], [], []
    g_pred, g_pnl, g_mask = [], [], []
    e_pred, e_pnl, e_mask = [], [], []

    with torch.no_grad():
        for x, static_adj, dynamic_adj, y_rank, y_pnl, is_event, trade_mask in loader:
            pred = model(x.to(device), static_adj.to(device), dynamic_adj.to(device)).detach().cpu().numpy().reshape(-1)
            rank = y_rank.numpy().reshape(-1)
            pnl = y_pnl.numpy().reshape(-1)
            mask = trade_mask.numpy().reshape(-1)
            event_flag = int(is_event.item())

            ic_h3 = safe_spearman_ic(pred, rank, mask)
            ic_t1 = safe_spearman_ic(pred, pnl, mask)

            if ic_h3 is not None:
                g_h3.append(ic_h3)
                if event_flag:
                    e_h3.append(ic_h3)

            if ic_t1 is not None:
                g_t1.append(ic_t1)
                if event_flag:
                    e_t1.append(ic_t1)

            g_pred.append(pred)
            g_pnl.append(pnl)
            g_mask.append(mask)

            if event_flag:
                e_pred.append(pred)
                e_pnl.append(pnl)
                e_mask.append(mask)

    g_ic = float(np.mean(g_h3)) if g_h3 else 0.0
    e_ic = float(np.mean(e_h3)) if e_h3 else 0.0
    g_t1_mean = float(np.mean(g_t1)) if g_t1 else 0.0
    e_t1_mean = float(np.mean(e_t1)) if e_t1 else 0.0

    g_h3_pct = 100.0 * g_ic
    e_h3_pct = 100.0 * e_ic
    g_t1_pct = 100.0 * g_t1_mean
    e_t1_pct = 100.0 * e_t1_mean

    event_score = 0.5 * e_h3_pct + 0.5 * e_t1_pct

    g_port = portfolio_metrics(g_pred, g_pnl, g_mask, annualized_sharpe=True)
    e_port = portfolio_metrics(e_pred, e_pnl, e_mask, annualized_sharpe=False)

    return {
        "G_IC": g_ic,
        "E_IC": e_ic,
        "G_T1": g_t1_mean,
        "E_T1": e_t1_mean,
        "G_SP": g_port["Sharpe"],
        "E_SR": e_port["Sharpe"],
        "MDD": g_port["MDD"],
        "Calmar": g_port["Calmar"],
        "EventVal": event_score,
    }

