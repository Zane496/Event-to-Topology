# Event-to-Topology (E2T) — Source Code

## Project layout (same layout as the original E2T_code.zip)

```text
E2T_github/
├── data/                    # five *_all dataset folders + original get_feature_baostock.py
├── models/
│   ├── e2t.py              # E2T Full Model only
│   └── layers.py           # shared R-GCN layer
├── scripts/
│   ├── run_cev.sh
│   ├── run_inu.sh
│   ├── run_emd.sh
│   ├── run_brn.sh
│   ├── run_mro.sh
│   └── run_all.sh
├── upstream/                # original module names, updated all-announcement Step0–7
├── utils/
│   ├── __init__.py
│   ├── data.py             # daily features and adjusted graphs
│   ├── losses.py           # H3/T1 masked ranking loss
│   ├── metrics.py          # six portfolio/ranking metrics, no Turn output
│   └── train_utils.py      # training and checkpoint selection
├── main.py                 # main CLI, same entrypoint as old zip
├── train.py                # downsteam orchestration
├── check_data.py           # optional input validator
├── run_steps.py            # optional Step0–7 wrapper
├── requirements.txt
└── README.md
```

**Original-layout shell scripts remain the primary entrypoint**, e.g. `bash scripts/run_cev.sh`, `bash scripts/run_brn.sh`, or `bash scripts/run_all.sh`. The scripts use existing Step7 outputs if present, generate them only when missing, run `check_data.py`, then call `main.py`. You can also run `python main.py --dataset cev_all` directly.


**Source-code release:** this repository contains the E2T Full Model and upstream scripts for `cev_all`, `inu_all`, `emd_all`, `brn_all`, and `mro_all`. The 15 baseline CSVs (nodes, market features, and prior relations) **are included**, but Teacher/Student outputs, Step7 dynamic adjacency arrays, model checkpoints, and experiment results are **not** included. Place locally generated compatible files under the dataset directories as described below. The root `.gitignore` allows these 15 baseline CSVs and excludes other generated dataset artifacts and model checkpoints.

The original full local project contained approximately 1.65 GB of extracted experiment assets. Do not upload it wholesale or rename ablation/old data files as Full Model inputs.

- **Downstream Full Model only:** shared R-GCN; binary topology-change gate; H3/T1 dual-target ranking; Top-20 portfolio; five seeds 3407–3411. **No ablation groups.**
- **Six reported downstream metrics only:** `E_IC`, `E_SR`, `G_IC`, `G_SP`, `MDD`, `Calmar` (means and population standard deviations across seeds). The experimental **Turn metric is not output** in logs, training history, per-seed CSV or summary CSV.
- Portfolio transaction costs **still** use internally calculated changes in holdings: fee 0.05% plus slippage 0.10% per changed long position, as in the reference BRN Full Model. Removing the metric does not remove or alter the cost adjustment.
- The stock/market-data acquisition file `data/get_feature_baostock.py` is unchanged from the old ZIP; upstream Student LoRA training and walk-forward OOF logic are retained.

## 1. Baseline CSVs and additional generated data

**The three basic CSVs per dataset are included in this release. You can copy additional real `_all` output files into this repository.** They must correspond to the *same actual stock universe*, have the expected fields and filenames, and not be old significant-event or `_199` outputs renamed with an `_all` suffix. You do not need to re-run the API, Step0–7, or Student training **when all required outputs already exist and are compatible**.

Files belong under `E2T_github/data/<dataset>/` (not `datasets/<dataset>/`). Example for `brn_all`:

```text
E2T_github/
  data/
    brn_all/
      brn_all_nodes.csv
      brn_all_features.csv
      brn_all_multi_relational_zero_shot_edges.csv
      llm_all_events_brn_all.csv             # if continuing from Step2
      ...                                     # other Step2–Step6 products, if present
      test_results_raw_train_OOF_Bstyle_compact_brn_all.csv
      test_results_raw_val_B_compact_brn_all.csv
      test_results_raw_test_B_compact_brn_all.csv
      downstream_inputs_brn_all/
        dynamic_4rel_graphs_brn_all.npy
        graph_dates_brn_all.csv
        node_index_brn_all.csv                # strongly recommended to verify exact node order
```

Apply the same filename pattern to `cev_all`, `inu_all`, `emd_all`, and `mro_all`.

**Minimum files needed to run downstream Full Model directly** are the three `*_nodes.csv`, `*_features.csv`, `*_multi_relational_zero_shot_edges.csv` files and the two Step7 outputs `dynamic_4rel_graphs_*.npy` and `graph_dates_*.csv` in `downstream_inputs_*`.

Important: Copy the corresponding **`node_index_<dataset>.csv`** from Step7 too when available. The node ordering in an adjacency tensor must exactly match the `*_nodes.csv` file. The provided input checker and downstream loader reject a mismatch when the index file is present. Without the index file, shape checks alone cannot detect a permuted same-size universe.

```bash
# from E2T_github/; check copied data WITHOUT training:
python check_data.py --dataset brn_all --stage downstream
python main.py --dataset brn_all

# all five datasets after copying all their matching inputs:
bash scripts/run_all.sh
```

**Reuse at an intermediate stage:** To resume from Step2, provide `llm_all_events_<dataset>.csv` and required nodes/prior files; to resume from Step7, provide the three Student prediction CSVs listed above and the nodes/prior CSVs. **Do not rename** old significant-event files into full-announcement inputs.

## 2. Configure API keys (only if you need to regenerate upstream data)

Do not paste a real API token into any `.py`, README, Git commit, or shared archive. The current code reads environment variables:

| Environment variable | Required for | Provider |
|---|---|---|
| `DASHSCOPE_API_KEY` (or fallback `QWEN_API_KEY`) | Step0 static prior graph, Step4 / Step6 Teacher API | Alibaba Cloud DashScope / Qwen |
| `TUSHARE_TOKEN` | Step1 earnings-forecast download **if an existing forecast cache is absent** | Tushare Pro |

For **Windows PowerShell**, replace placeholders with your own token before running the required steps in the *same terminal*:

```powershell
$env:DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_KEY"
$env:TUSHARE_TOKEN = "YOUR_TUSHARE_TOKEN"
```

For **Windows CMD**:

```bat
set "DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY"
set "TUSHARE_TOKEN=YOUR_TUSHARE_TOKEN"
```

For **Linux / macOS bash**:

```bash
export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_KEY'
export TUSHARE_TOKEN='YOUR_TUSHARE_TOKEN'
```

The example strings above are **placeholders, not usable keys**. You only need to set variables for operations that access the providers. **Direct Full Model training from copied Step7 output files does not require either API key.** If you already have the Step1 forecast cache, Step1 can use it without `TUSHARE_TOKEN`.

**Security reminder:** Some older scripts you previously supplied contained hard-coded API credentials. Revoke/rotate those credentials in the provider console before running, and keep the new ones outside source control.

## 3. Full upstream pipeline if you need to rebuild

Run from `E2T_github/` (substitute the dataset tag):

```bash
python run_steps.py --dataset brn_all --step 0         # creates prior graph; skip if already available
python check_data.py --dataset brn_all --stage base   # requires real nodes/features/prior
python run_steps.py --dataset brn_all --step 1         # creates llm_all_events_brn_all.csv; skip if available
python run_steps.py --dataset brn_all --step 2         # chronological train/val/test split
python run_steps.py --dataset brn_all --step 3         # Teacher/Student prompts
python run_steps.py --dataset brn_all --step 4         # Teacher labels for train + val; API calls
python run_steps.py --dataset brn_all --step 5         # SFT pairing
python run_steps.py --dataset brn_all --step 6 --split val  # val gold from Step4 answered output
python upstream/train_student.py --dataset brn_all     # GPU + Unsloth; original LoRA settings
python upstream/generate_oof.py --dataset brn_all      # 2022 none, 2023 <=2022, 2024 <=2023; val/test inference
python run_steps.py --dataset brn_all --step 7         # produces only Full-Model topology inputs
python check_data.py --dataset brn_all --stage downstream
python main.py --dataset brn_all
```

**Optional:** `python run_steps.py --dataset brn_all --step 6 --split test` requires a previously answered test Teacher file; the underlying script supports `--annotate`, and `run_steps.py` automatically annotates the test split when it invokes Step6. This is only needed if you explicitly want independent upstream test reference labels; the downstream Full Model does not need Teacher-generated test gold labels, and the Teacher must not be used for its test-time inference.

**Important order:** `check_data.py --stage base` requires an *existing* prior CSV. If you have nodes/features only, run Step0 first, then check. Avoid re-running completed steps: Teacher API calls incur cost, and regeneration can overwrite cached intermediates.

Dependencies are listed in `requirements.txt`; Student training/OOF requires a compatible CUDA, PyTorch and Unsloth installation. The `data/get_feature_baostock.py` original acquisition script still references the older dataset identifiers. It is left unmodified by request; prepare the five real `_all` node/feature CSVs separately and copy them in.

## 4. Full Model output layout

```text
data/brn_all/downstream_results_brn_all/
  checkpoints/full_model_seed_3407.pt
  full_model_training_seed_3407.csv
  full_model_per_seed_partial_brn_all.csv
  full_model_per_seed_brn_all.csv
  full_model_summary_brn_all.csv
```

Every other dataset follows the same pattern. Only six reported metrics are saved:

`E_IC`, `E_SR`, `G_IC`, `G_SP`, `MDD`, `Calmar`.

Training checkpoint selection uses the validation average of H3 and T1 event-day Spearman IC. Output files also include epochs, losses, selection score, and T1 diagnostic ICs; these are not additional portfolio metrics.

## 5. Scope of validation

The source archive contains code plus **15 baseline CSVs across five datasets**, but does **not** include the Step7 dynamic graphs or Student/OOF prediction files required to train immediately. Python syntax, file interfaces and light-weight smoke tests can be checked locally, but exact paper numbers require your actual five datasets and upstream Student outputs. No API calls or full LLM training are performed to produce this source package.

## 6. Windows quick start and GitHub publication

On Windows **CMD / PowerShell**, run the Python entrypoint from the inner `E2T_github` root (the directory containing `main.py`):

```powershell
cd D:\Pythonpro\machine\LLM_LoRA\E2T_github_920\E2T_github
python check_data.py --dataset cev_all --stage downstream
python main.py --dataset cev_all --seeds 3407
```

`*.sh` launchers require a Bash terminal (such as Git Bash, WSL or Linux); they are not directly executable in Windows CMD. In Git Bash, `bash scripts/run_all.sh` runs all five datasets. For the Python interpreter in your current environment use `python main.py ...` directly.

The public GitHub repository should contain only source code, README, dependencies and `.gitignore` (plus optional license). Dataset subdirectories, generated tensors, SFT labels, predictions and checkpoints are intentionally ignored. To publish a repository from your own account:

```bash
git init
git add .
git status
git commit -m "Initial E2T source release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/E2T.git
git push -u origin main
```

Create the **empty** repository on GitHub before running `git remote add origin`, leaving its automatic README / license initialization unchecked. GitHub authentication must be completed through your own authorized login (browser / credential manager / SSH); do **not** embed access tokens in source or remote URLs. If code files or keys have previously been committed, merely adding `.gitignore` is not sufficient to erase them from Git history.

The repository owner should choose and add a license appropriate for the paper's release. The included `requirements.txt` records the reference environment; CUDA / PyTorch / Unsloth may need platform-specific installation rather than a one-command Windows installation.


## 8. Important: what `bash scripts/run_all.sh` does with only the baseline CSVs

The fifteen included baseline CSVs are necessary inputs, **not sufficient inputs for downstream training**. The scripts perform the following steps per dataset:

1. If both `downstream_inputs_<dataset>/dynamic_4rel_graphs_<dataset>.npy` and `graph_dates_<dataset>.csv` already exist, reuse them.
2. Otherwise, try Step7 only if these three **existing Student outputs** are present in `data/<dataset>/`:
   - `test_results_raw_train_OOF_Bstyle_compact_<dataset>.csv`
   - `test_results_raw_val_B_compact_<dataset>.csv`
   - `test_results_raw_test_B_compact_<dataset>.csv`
3. Validate the inputs with `check_data.py`; train the Full Model using `main.py`.

The `.sh` scripts do **not** generate Teacher labels, train the LoRA Student, or run OOF inference automatically. If you have only the fifteen included baseline CSVs, **one-click downstream training is not yet possible**. Copy your already-generated Step7 and node-index files (or the three Student predictions), then run `bash scripts/run_all.sh` in Git Bash/WSL/Linux. For Windows CMD, use a Bash terminal or run `python main.py --dataset <dataset>` after the required files are present.

The market-data and relation CSVs may be subject to provider usage and redistribution terms. Before making this bundle public on GitHub, confirm you have permission to redistribute the feature and graph data; if uncertain, distribute the code-only edition instead and provide data preparation instructions.
