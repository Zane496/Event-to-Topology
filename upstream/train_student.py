"""Train and select the E2T Student with validation-only model selection.

The model, prompts, LoRA parameters, random seed, optimizer, early stopping,
inference, evaluation, and selection score are preserved from the original
experimental script. Console output is quiet by default; set ``VERBOSE=True``
for the original diagnostic messages.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import unsloth
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
MAX_SEQ_LENGTH = 3072
BASE_MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def set_global_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(3407)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the E2T Student model.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def configure_paths(dataset: str) -> None:
    global PROJECT_ROOT, BASE_DIR, DATA_DIR, CHECKPOINT_DIR
    global TRAIN_DATA, VAL_DATA, VAL_INPUT_CSV, VAL_GOLD_TRUTH_CSV
    global VAL_MODEL_SELECTION_CSV, VAL_MODEL_SELECTION_SORTED_CSV
    global LEGACY_VAL_MODEL_SELECTION_CSV, BEST_VAL_JSON

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    data_dir = PROJECT_ROOT / "data" / dataset
    checkpoint_dir = data_dir / "checkpoints"

    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    DATA_DIR = str(data_dir)
    BASE_DIR = DATA_DIR
    CHECKPOINT_DIR = str(checkpoint_dir)

    TRAIN_DATA = str(
        data_dir / f"sft_data_train_compact_{dataset}.jsonl"
    )
    VAL_DATA = str(
        data_dir / f"sft_data_val_compact_{dataset}.jsonl"
    )
    VAL_INPUT_CSV = str(
        data_dir / f"llm_exam_val_student_{dataset}.csv"
    )
    VAL_GOLD_TRUTH_CSV = str(
        data_dir / f"qwen_max_val_gold_edges_{dataset}.csv"
    )

    VAL_MODEL_SELECTION_CSV = str(
        checkpoint_dir
        / f"val_model_selection_results_compact_{dataset}.csv"
    )
    VAL_MODEL_SELECTION_SORTED_CSV = str(
        checkpoint_dir
        / f"val_model_selection_leaderboard_sorted_{dataset}.csv"
    )
    BEST_VAL_JSON = str(
        checkpoint_dir
        / f"best_lora_selected_by_val_{dataset}.json"
    )

    LEGACY_VAL_MODEL_SELECTION_CSV = str(
        data_dir
        / f"VAL_MODEL_SELECTION_RESULTS_COMPACT_{dataset.upper()}.csv"
    )

configure_paths(DEFAULT_DATASET)

SYSTEM_PROMPT = """你是金融事件标签器。只输出JSON数组，不输出解释。
每个候选输出[i,a,d,s,p]。
i=候选编号。
a=P维持/D弱化/I无关。
d=-1负向/0中性/1正向。
s=冲击强度0-3。
p=置信度0-3。
禁止输出候选编号以外的项目。
禁止输出股票代码、公司名、reason或解释文字。"""


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]

    texts = []

    for instruction, output in zip(instructions, outputs):
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )

        texts.append(text)

    return {"text": texts}


def check_required_files():
    """Check only the files required for training and validation.

    Test data must not be accessed during model selection.
    """
    required_files = [
        TRAIN_DATA,
        VAL_DATA,
        VAL_INPUT_CSV,
        VAL_GOLD_TRUTH_CSV,
    ]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required training/validation files are missing:\n" + missing_text
        )



def run_training(config):
    set_global_seed(3407)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config["lora_alpha"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset_train = load_dataset(
        "json",
        data_files=TRAIN_DATA,
        split="train",
    )

    dataset_val = load_dataset(
        "json",
        data_files=VAL_DATA,
        split="train",
    )

    dataset_train = dataset_train.shuffle(seed=3407).map(
        formatting_prompts_func,
        batched=True,
    )

    dataset_val = dataset_val.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            output_dir=os.path.join(DATA_DIR, f"outputs_{config['run_name']}"),
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["grad_acc_steps"],
            num_train_epochs=config["epochs"],
            warmup_ratio=0.1,
            learning_rate=config["learning_rate"],
            lr_scheduler_type="cosine",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            report_to="none",
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            )
        ],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    save_path = os.path.join(DATA_DIR, config["save_name"])

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return save_path



def count_candidates_from_prompt(prompt_text):
    """Count candidate rows in the Student prompt.

    Expected row format:
        1|002197|Company_A|COMPETITOR
        2|300209|Company_B|SUPPLY_CHAIN
    """
    return len(
        re.findall(
            r"^\s*\d+\|",
            str(prompt_text),
            flags=re.MULTILINE,
        )
    )


def dynamic_max_new_tokens(prompt_text):
    n = count_candidates_from_prompt(prompt_text)

    if n <= 0:
        n = 8

    return min(
        512,
        max(128, 32 * n + 64)
    )


def run_inference(lora_path, test_csv, output_csv):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    df_test = pd.read_csv(test_csv, dtype={"Symbol": str})
    df_test["Symbol"] = df_test["Symbol"].astype(str).str.zfill(6)

    results = []

    for _, row in tqdm(
        df_test.iterrows(),
        total=len(df_test),
        desc="Student inference",
    ):
        instruction = str(row["Student_Prompt"])

        prompt_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to("cuda")

        max_new_tokens = dynamic_max_new_tokens(instruction)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][len(inputs.input_ids[0]):]
        pred_response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        pred_response = pred_response.replace("```json", "")
        pred_response = pred_response.replace("```", "")
        pred_response = pred_response.strip()

        results.append({
            "Date": row.get("Date", ""),
            "Symbol": str(row.get("Symbol", "")).zfill(6),
            "Name": row.get("Name", ""),
            "Candidates_JSON": row.get("Candidates_JSON", ""),
            "Prompt": instruction,
            "Raw_Output": pred_response,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf_8_sig")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()



def extract_json_array_robust(text):
    try:
        text = str(text).strip()

        left = text.find("[")
        right = text.rfind("]")

        if left != -1 and right != -1 and right > left:
            return json.loads(text[left:right + 1])

        match = re.search(r"\[.*\]", text, re.DOTALL)

        if match:
            return json.loads(match.group(0))

        return None

    except Exception:
        return None


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_action_code(x):
    x = str(x).strip().upper()

    if x in {"P", "PRESERVED"}:
        return "P"

    if x in {"D", "DECOUPLED"}:
        return "D"

    if x in {"I", "ISOLATED"}:
        return "I"

    return "INVALID"


def clip_bucket_0_3(x):
    v = safe_int(x, default=0)
    return int(max(0, min(3, v)))


def parse_student_compact_output(raw_output):
    """Parse compact Student output in ``[[i, a, d, s, p], ...]`` format.

    Returns one normalized record per candidate.
    """
    parsed = extract_json_array_robust(raw_output)

    if not isinstance(parsed, list):
        return None

    results = []

    for item in parsed:
        if not isinstance(item, list):
            return None

        if len(item) < 5:
            return None

        cid = safe_int(item[0])
        action_code = normalize_action_code(item[1])
        direction = safe_int(item[2])
        strength_bucket = clip_bucket_0_3(item[3])
        confidence_bucket = clip_bucket_0_3(item[4])

        if cid <= 0:
            return None

        if action_code not in {"P", "D", "I"}:
            return None

        if direction not in {-1, 0, 1}:
            direction = 0

        results.append({
            "Candidate_ID": cid,
            "Pred_Action_Code": action_code,
            "Pred_Direction": direction,
            "Pred_Strength_Bucket": strength_bucket,
            "Pred_Confidence_Bucket": confidence_bucket,
        })

    return results



def evaluate_performance(pred_file, gold_file, output_csv):
    try:
        df_pred = pd.read_csv(pred_file, dtype={"Symbol": str})
    except FileNotFoundError:
        print(f"Warning: prediction file not found: {pred_file}")
        return None

    try:
        df_gold = pd.read_csv(
            gold_file,
            dtype={
                "Symbol": str,
                "Target_Code": str,
            },
        )
    except FileNotFoundError:
        print(f"Warning: gold-label file not found: {gold_file}")
        return None

    df_pred["Symbol"] = df_pred["Symbol"].astype(str).str.zfill(6)
    df_gold["Symbol"] = df_gold["Symbol"].astype(str).str.zfill(6)

    df_gold["Date"] = df_gold["Date"].astype(str)
    df_pred["Date"] = df_pred["Date"].astype(str)

    df_gold["Candidate_ID"] = df_gold["Candidate_ID"].astype(int)

    if "True_Action_Code" not in df_gold.columns:
        action_map = {
            "PRESERVED": "P",
            "DECOUPLED": "D",
            "ISOLATED": "I",
        }
        df_gold["True_Action_Code"] = df_gold["True_Action"].map(action_map)

    df_gold["True_Direction"] = df_gold["True_Direction"].astype(int)
    df_gold["True_Strength_Bucket"] = df_gold["True_Strength_Bucket"].astype(int)
    df_gold["True_Confidence_Bucket"] = df_gold["True_Confidence_Bucket"].astype(int)

    gold_group = {
        (str(k[0]), str(k[1]).zfill(6)): set(v["Candidate_ID"].astype(int).tolist())
        for k, v in df_gold.groupby(["Date", "Symbol"])
    }

    parsed_rows = []

    total_events = len(df_pred)
    parse_success_events = 0
    duplicate_candidate_count = 0
    invalid_candidate_count = 0
    parsed_edge_count = 0

    for _, row in df_pred.iterrows():
        date = str(row["Date"]).strip()
        symbol = str(row["Symbol"]).zfill(6)
        raw_output = str(row.get("Raw_Output", ""))

        parsed = parse_student_compact_output(raw_output)

        if parsed is None:
            continue

        parse_success_events += 1

        valid_ids = gold_group.get((date, symbol), set())
        seen_ids = set()

        for item in parsed:
            cid = int(item["Candidate_ID"])

            if cid in seen_ids:
                duplicate_candidate_count += 1
                continue

            seen_ids.add(cid)

            if cid not in valid_ids:
                invalid_candidate_count += 1
                continue

            parsed_edge_count += 1

            parsed_rows.append({
                "Date": date,
                "Symbol": symbol,
                "Candidate_ID": cid,
                "Pred_Action_Code": item["Pred_Action_Code"],
                "Pred_Direction": item["Pred_Direction"],
                "Pred_Strength_Bucket": item["Pred_Strength_Bucket"],
                "Pred_Confidence_Bucket": item["Pred_Confidence_Bucket"],
            })

    parse_success_rate = parse_success_events / max(total_events, 1)

    df_parsed = pd.DataFrame(parsed_rows)

    if df_parsed.empty:
        print("Warning: no valid Student predictions were parsed.")
        return None

    df_eval = pd.merge(
        df_gold,
        df_parsed,
        on=["Date", "Symbol", "Candidate_ID"],
        how="left",
    )

    df_eval["Pred_Action_Code"] = df_eval["Pred_Action_Code"].fillna("MISSING")
    df_eval["Pred_Direction"] = df_eval["Pred_Direction"].fillna(99).astype(int)
    df_eval["Pred_Strength_Bucket"] = df_eval["Pred_Strength_Bucket"].fillna(99).astype(int)
    df_eval["Pred_Confidence_Bucket"] = df_eval["Pred_Confidence_Bucket"].fillna(99).astype(int)

    total_edges = len(df_eval)
    covered_edges = (df_eval["Pred_Action_Code"] != "MISSING").sum()
    coverage_rate = covered_edges / max(total_edges, 1)

    action_acc = accuracy_score(
        df_eval["True_Action_Code"],
        df_eval["Pred_Action_Code"],
    )

    action_macro_f1 = f1_score(
        df_eval["True_Action_Code"],
        df_eval["Pred_Action_Code"],
        labels=["P", "D", "I"],
        average="macro",
        zero_division=0,
    )

    direction_acc = accuracy_score(
        df_eval["True_Direction"],
        df_eval["Pred_Direction"],
    )

    direction_macro_f1 = f1_score(
        df_eval["True_Direction"],
        df_eval["Pred_Direction"],
        labels=[-1, 0, 1],
        average="macro",
        zero_division=0,
    )

    strength_acc = accuracy_score(
        df_eval["True_Strength_Bucket"],
        df_eval["Pred_Strength_Bucket"],
    )

    confidence_acc = accuracy_score(
        df_eval["True_Confidence_Bucket"],
        df_eval["Pred_Confidence_Bucket"],
    )

    valid_strength_mask = df_eval["Pred_Strength_Bucket"].between(0, 3)
    valid_conf_mask = df_eval["Pred_Confidence_Bucket"].between(0, 3)

    if valid_strength_mask.sum() > 0:
        strength_abs_err = (
            df_eval.loc[valid_strength_mask, "True_Strength_Bucket"]
            - df_eval.loc[valid_strength_mask, "Pred_Strength_Bucket"]
        ).abs()

        strength_mae = strength_abs_err.mean()
        strength_within1 = (strength_abs_err <= 1).mean()
    else:
        strength_mae = np.nan
        strength_within1 = 0.0

    if valid_conf_mask.sum() > 0:
        confidence_abs_err = (
            df_eval.loc[valid_conf_mask, "True_Confidence_Bucket"]
            - df_eval.loc[valid_conf_mask, "Pred_Confidence_Bucket"]
        ).abs()

        confidence_mae = confidence_abs_err.mean()
        confidence_within1 = (confidence_abs_err <= 1).mean()
    else:
        confidence_mae = np.nan
        confidence_within1 = 0.0

    action_direction_joint = (
        (df_eval["True_Action_Code"] == df_eval["Pred_Action_Code"])
        & (df_eval["True_Direction"] == df_eval["Pred_Direction"])
    ).mean()

    action_direction_strength_joint = (
        (df_eval["True_Action_Code"] == df_eval["Pred_Action_Code"])
        & (df_eval["True_Direction"] == df_eval["Pred_Direction"])
        & (df_eval["True_Strength_Bucket"] == df_eval["Pred_Strength_Bucket"])
    ).mean()

    full_joint = (
        (df_eval["True_Action_Code"] == df_eval["Pred_Action_Code"])
        & (df_eval["True_Direction"] == df_eval["Pred_Direction"])
        & (df_eval["True_Strength_Bucket"] == df_eval["Pred_Strength_Bucket"])
        & (df_eval["True_Confidence_Bucket"] == df_eval["Pred_Confidence_Bucket"])
    ).mean()

    student_score = (
        0.35 * action_macro_f1
        + 0.35 * direction_macro_f1
        + 0.15 * strength_within1
        + 0.15 * confidence_within1
    )

    df_eval.to_csv(output_csv, index=False, encoding="utf_8_sig")


    return {
        "Parse_Success": parse_success_rate,
        "Coverage": coverage_rate,
        "Duplicate_Count": duplicate_candidate_count,
        "Invalid_ID_Count": invalid_candidate_count,

        "Action_Acc": action_acc,
        "Action_Macro_F1": action_macro_f1,

        "Direction_Acc": direction_acc,
        "Direction_Macro_F1": direction_macro_f1,

        "Strength_Acc": strength_acc,
        "Strength_MAE": strength_mae,
        "Strength_Within1": strength_within1,

        "Confidence_Acc": confidence_acc,
        "Confidence_MAE": confidence_mae,
        "Confidence_Within1": confidence_within1,

        "Action_Direction_Joint": action_direction_joint,
        "Action_Direction_Strength_Joint": action_direction_strength_joint,
        "Full_Joint": full_joint,

        "StudentScore": student_score,
    }

def save_text_report(config, metrics, lora_weights_path, pred_output_csv, eval_output_csv):
    txt_report_path = os.path.join(DATA_DIR, f"val_report_{config['run_name']}.txt")

    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Run: {config['run_name']} - validation report\n")
        f.write("=" * 60 + "\n")

        f.write("\n[Hyperparameters]\n")
        f.write(f" - LoRA Rank: {config['r']}\n")
        f.write(f" - LoRA Alpha: {config['lora_alpha']}\n")
        f.write(f" - Learning rate: {config['learning_rate']}\n")
        f.write(f" - Epochs: {config['epochs']}\n")
        f.write(f" - Batch size: {config['batch_size']}\n")
        f.write(f" - Gradient accumulation: {config['grad_acc_steps']}\n")
        f.write(f" - Early-stopping patience: {config['early_stopping_patience']}\n")

        f.write("\n[Format and coverage]\n")
        f.write(f" - Parse Success: {metrics['Parse_Success']:.2%}\n")
        f.write(f" - Coverage: {metrics['Coverage']:.2%}\n")
        f.write(f" - Duplicate Count: {metrics['Duplicate_Count']}\n")
        f.write(f" - Invalid ID Count: {metrics['Invalid_ID_Count']}\n")

        f.write("\n[Action]\n")
        f.write(f" - Action Acc: {metrics['Action_Acc']:.4f}\n")
        f.write(f" - Action Macro-F1: {metrics['Action_Macro_F1']:.4f}\n")

        f.write("\n[Direction]\n")
        f.write(f" - Direction Acc: {metrics['Direction_Acc']:.4f}\n")
        f.write(f" - Direction Macro-F1: {metrics['Direction_Macro_F1']:.4f}\n")

        f.write("\n[Strength bucket]\n")
        f.write(f" - Strength Acc: {metrics['Strength_Acc']:.4f}\n")
        f.write(f" - Strength MAE: {metrics['Strength_MAE']:.4f}\n")
        f.write(f" - Strength Within-1: {metrics['Strength_Within1']:.4f}\n")

        f.write("\n[Confidence bucket]\n")
        f.write(f" - Confidence Acc: {metrics['Confidence_Acc']:.4f}\n")
        f.write(f" - Confidence MAE: {metrics['Confidence_MAE']:.4f}\n")
        f.write(f" - Confidence Within-1: {metrics['Confidence_Within1']:.4f}\n")

        f.write("\n[Joint metrics]\n")
        f.write(f" - Action+Direction Joint: {metrics['Action_Direction_Joint']:.4f}\n")
        f.write(
            f" - Action+Direction+Strength Joint: "
            f"{metrics['Action_Direction_Strength_Joint']:.4f}\n"
        )
        f.write(f" - Full Joint: {metrics['Full_Joint']:.4f}\n")

        f.write("\n[Composite score]\n")
        f.write(f" - StudentScore: {metrics['StudentScore']:.4f}\n")

        f.write("\n[Artifacts]\n")
        f.write(f" - Model weights: {lora_weights_path}\n")
        f.write(f" - Predictions: {pred_output_csv}\n")
        f.write(f" - Evaluation details: {eval_output_csv}\n")
        f.write("=" * 60 + "\n")



def to_repository_relative_path(path_value) -> str:
    """Return a portable repository-relative artifact path when possible."""
    path = Path(path_value)

    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_result_row(config, metrics, lora_weights_path):
    return {
        "Finished_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Run_Name": config["run_name"],
        "Save_Path": to_repository_relative_path(lora_weights_path),
        "r": config["r"],
        "alpha": config["lora_alpha"],
        "lr": config["learning_rate"],
        "Epochs": config["epochs"],

        "Parse_Success": f"{metrics['Parse_Success']:.2%}",
        "Coverage": f"{metrics['Coverage']:.2%}",
        "Duplicate_Count": metrics["Duplicate_Count"],
        "Invalid_ID_Count": metrics["Invalid_ID_Count"],

        "Action_Acc": f"{metrics['Action_Acc']:.4f}",
        "Action_Macro_F1": f"{metrics['Action_Macro_F1']:.4f}",

        "Direction_Acc": f"{metrics['Direction_Acc']:.4f}",
        "Direction_Macro_F1": f"{metrics['Direction_Macro_F1']:.4f}",

        "Strength_Acc": f"{metrics['Strength_Acc']:.4f}",
        "Strength_MAE": f"{metrics['Strength_MAE']:.4f}",
        "Strength_Within1": f"{metrics['Strength_Within1']:.4f}",

        "Confidence_Acc": f"{metrics['Confidence_Acc']:.4f}",
        "Confidence_MAE": f"{metrics['Confidence_MAE']:.4f}",
        "Confidence_Within1": f"{metrics['Confidence_Within1']:.4f}",

        "Action_Direction_Joint": f"{metrics['Action_Direction_Joint']:.4f}",
        "Action_Direction_Strength_Joint": f"{metrics['Action_Direction_Strength_Joint']:.4f}",
        "Full_Joint": f"{metrics['Full_Joint']:.4f}",

        "StudentScore": f"{metrics['StudentScore']:.4f}",
    }


def append_result_row_to_csv(result_row, output_path):
    """Append one completed configuration without overwriting prior results."""
    row_df = pd.DataFrame([result_row])

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    row_df.to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf_8_sig",
    )


def _metric_to_float(series):
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )


def load_val_model_selection_history():
    """Load validation model-selection history.

    The legacy uppercase file is read first, followed by the current append-only
    file. When a run name appears more than once, the latest row is retained.
    """
    frames = []

    for path in [LEGACY_VAL_MODEL_SELECTION_CSV, VAL_MODEL_SELECTION_CSV]:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path, dtype=str)
            if not df.empty:
                df["_Source_CSV"] = path
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    if "Run_Name" not in df_all.columns:
        raise ValueError("Validation model-selection CSV is missing the Run_Name column.")

    # Retain the latest row when the same configuration is rerun.
    df_all = df_all.drop_duplicates(subset=["Run_Name"], keep="last").reset_index(drop=True)

    return df_all


def save_best_from_val_history(hyperparameter_configs):
    """Select the global best completed configuration by validation StudentScore."""
    df_master = load_val_model_selection_history()

    if df_master.empty:
        return None

    required_cols = ["Run_Name", "Save_Path", "StudentScore"]
    missing_cols = [col for col in required_cols if col not in df_master.columns]
    if missing_cols:
        raise ValueError(f"Validation model-selection CSV is missing columns: {missing_cols}")

    df_master["_StudentScore_Float"] = _metric_to_float(df_master["StudentScore"])

    if "Action_Macro_F1" in df_master.columns:
        df_master["_Action_Macro_F1_Float"] = _metric_to_float(df_master["Action_Macro_F1"])
    else:
        df_master["_Action_Macro_F1_Float"] = 0.0

    if "Direction_Macro_F1" in df_master.columns:
        df_master["_Direction_Macro_F1_Float"] = _metric_to_float(df_master["Direction_Macro_F1"])
    else:
        df_master["_Direction_Macro_F1_Float"] = 0.0

    df_master = df_master.sort_values(
        by=[
            "_StudentScore_Float",
            "_Action_Macro_F1_Float",
            "_Direction_Macro_F1_Float",
        ],
        ascending=False,
    ).reset_index(drop=True)

    save_cols = [col for col in df_master.columns if not col.startswith("_")]
    df_master[save_cols].to_csv(
        VAL_MODEL_SELECTION_SORTED_CSV,
        index=False,
        encoding="utf_8_sig",
    )

    best_row = df_master.iloc[0]
    best_run_name = str(best_row["Run_Name"])
    best_config = next(
        (c for c in hyperparameter_configs if c["run_name"] == best_run_name),
        None,
    )

    best_record = {
        "dataset": Path(DATA_DIR).name,
        "method": "E2T",
        "checkpoint_role": "final_student_for_validation_and_test",
        "selection_split": "validation",
        "selection_metric": "StudentScore",
        "run_name": best_run_name,
        "lora_path": to_repository_relative_path(
            best_row["Save_Path"]
        ),
        "checkpoint_included_in_archive": False,
        "checkpoint_generation_command": (
            f"python upstream/train_student.py --dataset {dataset}"
        ),
        "released_artifact": "frozen_predictions",
        "base_model": BASE_MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "seed": 3407,
        "val_student_score": float(
            best_row["_StudentScore_Float"]
        ),
        "val_action_macro_f1": float(
            best_row["_Action_Macro_F1_Float"]
        ),
        "val_direction_macro_f1": float(
            best_row["_Direction_Macro_F1_Float"]
        ),
        "config": best_config,
        "leaderboard_append_csv": (
            to_repository_relative_path(
                VAL_MODEL_SELECTION_CSV
            )
        ),
        "leaderboard_sorted_csv": (
            to_repository_relative_path(
                VAL_MODEL_SELECTION_SORTED_CSV
            )
        ),
        "test_used_for_selection": False,
        "prompt_language": "Chinese",
        "notes": (
            "The final E2T Student was selected using validation data only. "
            "Frozen Train OOF predictions were generated by chronology-safe "
            "annual Students and cannot be reconstructed from this final "
            "LoRA alone."
        ),
    }

    with open(BEST_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(best_record, f, ensure_ascii=False, indent=2)


    return best_record


def train_student(dataset: str = DEFAULT_DATASET) -> dict | None:
    configure_paths(dataset)
    check_required_files()

    hyperparameter_configs = [
        {
            "run_name": f"config_A_baseline_{dataset}",
            "save_name": os.path.join("checkpoints", "e2t_student_lora"),
            "r": 16,
            "lora_alpha": 32,
            "learning_rate": 5e-5,
            "epochs": 16,
            "batch_size": 1,
            "grad_acc_steps": 8,
            "early_stopping_patience": 5,
        }
    ]

    for config in hyperparameter_configs:
        lora_weights_path = run_training(config)
        pred_output_csv = os.path.join(
            DATA_DIR,
            f"val_results_raw_{config['run_name']}.csv",
        )
        run_inference(
            lora_path=lora_weights_path,
            test_csv=VAL_INPUT_CSV,
            output_csv=pred_output_csv,
        )
        eval_output_csv = os.path.join(
            DATA_DIR,
            f"val_eval_results_{config['run_name']}.csv",
        )
        metrics = evaluate_performance(
            pred_file=pred_output_csv,
            gold_file=VAL_GOLD_TRUTH_CSV,
            output_csv=eval_output_csv,
        )
        if metrics:
            save_text_report(
                config=config,
                metrics=metrics,
                lora_weights_path=lora_weights_path,
                pred_output_csv=pred_output_csv,
                eval_output_csv=eval_output_csv,
            )
            result_row = build_result_row(config, metrics, lora_weights_path)
            append_result_row_to_csv(result_row, VAL_MODEL_SELECTION_CSV)
            save_best_from_val_history(hyperparameter_configs)

    best = save_best_from_val_history(hyperparameter_configs)
    if best is None:
        raise RuntimeError("No valid Student checkpoint was selected on validation data.")
    print(
        f"Selected {best['run_name']} for {dataset} "
        f"(Val StudentScore={best['val_student_score']:.4f})"
    )
    return best


def main() -> None:
    args = parse_args()
    train_student(args.dataset)


if __name__ == "__main__":
    main()
