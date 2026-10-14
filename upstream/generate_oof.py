"""Generate chronology-safe train OOF and validation/test Student outputs.

The annual walk-forward blocks, LoRA configuration, random seed, data split,
training objective, and greedy decoding settings are preserved from the
original experiment. Manual console logs are quiet by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import warnings
from pathlib import Path

import unsloth
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

DEFAULT_DATASET = "brn_all"
SUPPORTED_DATASETS = ("cev_all", "inu_all", "emd_all", "brn_all", "mro_all")
BASE_MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 3072
DELETE_TEMP_LORA_AFTER_INFER = False
RUN_VAL_INFERENCE = True
RUN_TEST_INFERENCE = True
COPY_EXISTING_TEST_TO_STANDARD_NAME = False
TIME_BLOCKS = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

SYSTEM_PROMPT = """你是金融事件标签器。只输出JSON数组，不输出解释。
每个候选输出[i,a,d,s,p]。
i=候选编号。
a=P维持/D弱化/I无关。
d=-1负向/0中性/1正向。
s=冲击强度0-3。
p=置信度0-3。
禁止输出候选编号以外的项目。
禁止输出股票代码、公司名、reason或解释文字。"""


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
    parser = argparse.ArgumentParser(description="Generate E2T OOF topology signals.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=SUPPORTED_DATASETS,
    )
    return parser.parse_args()


def configure_paths(dataset: str) -> None:
    global BASE_DIR, DATA_DIR, BEST_FULL_LORA_PATH
    global RAW_TRAIN_JSONL, RAW_TRAIN_CSV, RAW_VAL_CSV, RAW_TEST_CSV
    global TEMP_TRAIN_JSONL_TPL, TEMP_EVAL_JSONL_TPL, TEMP_INFER_CSV_TPL
    global OOF_PRED_CSV_TPL, FINAL_OOF_SAFE_TRAIN_CSV
    global FINAL_VAL_PRED_CSV, FINAL_TEST_PRED_CSV

    DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / dataset)
    BASE_DIR = DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)

    BEST_FULL_LORA_PATH = os.path.join(
        DATA_DIR,
        "checkpoints",
        "e2t_student_lora",
    )
    RAW_TRAIN_JSONL = os.path.join(
        DATA_DIR,
        f"sft_data_train_compact_{dataset}.jsonl",
    )
    RAW_TRAIN_CSV = os.path.join(
        DATA_DIR,
        f"llm_exam_train_student_{dataset}.csv",
    )
    RAW_VAL_CSV = os.path.join(
        DATA_DIR,
        f"llm_exam_val_student_{dataset}.csv",
    )
    RAW_TEST_CSV = os.path.join(
        DATA_DIR,
        f"llm_exam_test_student_{dataset}.csv",
    )
    TEMP_TRAIN_JSONL_TPL = os.path.join(
        DATA_DIR,
        f"temp_oof_train_block_{{}}_{dataset}.jsonl",
    )
    TEMP_EVAL_JSONL_TPL = os.path.join(
        DATA_DIR,
        f"temp_oof_eval_block_{{}}_{dataset}.jsonl",
    )
    TEMP_INFER_CSV_TPL = os.path.join(
        DATA_DIR,
        f"temp_oof_infer_block_{{}}_{dataset}.csv",
    )
    OOF_PRED_CSV_TPL = os.path.join(
        DATA_DIR,
        f"oof_pred_block_{{}}_compact_Bstyle_{dataset}.csv",
    )
    FINAL_OOF_SAFE_TRAIN_CSV = os.path.join(
        DATA_DIR,
        f"test_results_raw_train_OOF_Bstyle_compact_{dataset}.csv",
    )
    FINAL_VAL_PRED_CSV = os.path.join(
        DATA_DIR,
        f"test_results_raw_val_B_compact_{dataset}.csv",
    )
    FINAL_TEST_PRED_CSV = os.path.join(
        DATA_DIR,
        f"test_results_raw_test_B_compact_{dataset}.csv",
    )


configure_paths(DEFAULT_DATASET)

def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)


def check_required_files():
    required = [
        RAW_TRAIN_JSONL,
        RAW_TRAIN_CSV,
        RAW_VAL_CSV,
    ]

    if RUN_TEST_INFERENCE:
        required.append(RAW_TEST_CSV)

    missing = [x for x in required if not os.path.exists(x)]

    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required OOF input files are missing:\n" + missing_text
        )

    if (RUN_VAL_INFERENCE or RUN_TEST_INFERENCE) and not os.path.exists(
        BEST_FULL_LORA_PATH
    ):
        print(
            "Warning: the validation-selected full Student checkpoint was not "
            f"found at {BEST_FULL_LORA_PATH}; validation/test inference will be skipped."
        )


def extract_date_from_instruction(instruction):
    text = str(instruction)

    patterns = [
        r"日期\s*=\s*(\d{4}-\d{2}-\d{2})",
        r"日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})",
        r"发布日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    return None


def read_jsonl_with_dates(jsonl_path):
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            event_date = extract_date_from_instruction(obj.get("instruction", ""))

            if event_date is None:
                continue

            obj["_event_date"] = event_date
            rows.append(obj)

    rows = sorted(rows, key=lambda x: x["_event_date"])

    return rows


def write_jsonl(rows, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in rows:
            clean_obj = {k: v for k, v in obj.items() if not k.startswith("_")}
            f.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")


def split_train_eval_from_past(past_rows, eval_ratio=0.2, min_eval=3):
    """Split training/evaluation data using past samples only.

    Future validation labels are never used for early stopping.
    """
    n = len(past_rows)

    if n == 0:
        return [], []

    if n < 10:
        return past_rows, past_rows

    eval_size = max(min_eval, int(round(n * eval_ratio)))
    eval_size = min(eval_size, max(1, n - 1))

    train_rows = past_rows[:-eval_size]
    eval_rows = past_rows[-eval_size:]

    if len(train_rows) == 0:
        train_rows = past_rows

    if len(eval_rows) == 0:
        eval_rows = past_rows[-1:]

    return train_rows, eval_rows


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


def count_candidates_from_prompt(prompt_text):
    """Count candidate rows in the Student prompt.

    Expected row format: ``1|002197|Company_A|COMPETITOR``.
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

    return min(512, max(128, 32 * n + 64))


def clean_raw_output(text):
    text = str(text).strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text.strip()


def get_date_col(df):
    if "Date" in df.columns:
        return "Date"
    if "日期" in df.columns:
        return "日期"
    raise KeyError("Neither the Date column nor its legacy Chinese alias was found.")


def get_symbol_col(df):
    if "Symbol" in df.columns:
        return "Symbol"
    if "代码" in df.columns:
        return "代码"
    raise KeyError("Neither the Symbol column nor its legacy Chinese alias was found.")


def get_prompt_col(df):
    if "Student_Prompt" in df.columns:
        return "Student_Prompt"
    if "Prompt" in df.columns:
        return "Prompt"
    raise KeyError("Neither Student_Prompt nor Prompt was found.")

def prepare_temporal_data(
    raw_csv_path,
    raw_jsonl_path,
    train_end_date,
    infer_start_date,
    infer_end_date,
    current_block,
):
    """Prepare past-only training data and the current inference block.

    Training rows satisfy ``event_date <= train_end_date``. Inference rows
    satisfy ``infer_start_date <= Date <= infer_end_date``.
    """
    all_rows = read_jsonl_with_dates(raw_jsonl_path)

    past_rows = [row for row in all_rows if row["_event_date"] <= train_end_date]

    train_rows, eval_rows = split_train_eval_from_past(past_rows)

    current_train_jsonl = TEMP_TRAIN_JSONL_TPL.format(current_block)
    current_eval_jsonl = TEMP_EVAL_JSONL_TPL.format(current_block)

    write_jsonl(train_rows, current_train_jsonl)
    write_jsonl(eval_rows, current_eval_jsonl)

    df_raw = pd.read_csv(raw_csv_path, dtype={"Symbol": str, "代码": str})
    date_col = get_date_col(df_raw)

    df_raw[date_col] = pd.to_datetime(df_raw[date_col])

    mask = (df_raw[date_col] >= pd.to_datetime(infer_start_date)) & (
        df_raw[date_col] <= pd.to_datetime(infer_end_date)
    )

    df_infer = df_raw[mask].copy()
    df_infer[date_col] = df_infer[date_col].dt.strftime("%Y-%m-%d")

    current_infer_csv = TEMP_INFER_CSV_TPL.format(current_block)
    df_infer.to_csv(current_infer_csv, index=False, encoding="utf_8_sig")


    return (
        current_train_jsonl,
        current_eval_jsonl,
        current_infer_csv,
        len(train_rows),
        len(df_infer),
    )


def run_training(config, train_data_path, eval_data_path):
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
        data_files=train_data_path,
        split="train",
    )

    dataset_eval = load_dataset(
        "json",
        data_files=eval_data_path,
        split="train",
    )

    dataset_train = dataset_train.shuffle(seed=3407).map(
        formatting_prompts_func,
        batched=True,
    )

    dataset_eval = dataset_eval.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            output_dir=os.path.join(BASE_DIR, f"outputs_{config['run_name']}"),
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
            save_total_limit=2,
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

    save_path = os.path.join(BASE_DIR, config["save_name"])

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return save_path

def run_inference(lora_path, input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Warning: inference input file not found; skipped: {input_csv}")
        return False

    df_test = pd.read_csv(input_csv, dtype={"Symbol": str, "代码": str})

    if df_test.empty:
        print(f"Warning: inference input file is empty; skipped: {input_csv}")
        return False

    date_col = get_date_col(df_test)
    symbol_col = get_symbol_col(df_test)
    prompt_col = get_prompt_col(df_test)

    df_test[symbol_col] = df_test[symbol_col].astype(str).str.zfill(6)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    results = []

    for _, row in tqdm(
        df_test.iterrows(),
        total=len(df_test),
        desc="Student inference",
    ):
        instruction = str(row[prompt_col])

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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=dynamic_max_new_tokens(instruction),
                use_cache=True,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][len(inputs.input_ids[0]) :]
        pred_response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        pred_response = clean_raw_output(pred_response)

        results.append(
            {
                "Date": row.get(date_col, ""),
                "Symbol": str(row.get(symbol_col, "")).zfill(6),
                "Name": row.get("Name", ""),
                "Candidates_JSON": row.get("Candidates_JSON", ""),
                "Student_Prompt": instruction,
                "Raw_Output": pred_response,
            }
        )

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf_8_sig")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return True

def merge_oof_predictions(oof_files, output_csv):
    valid_files = [f for f in oof_files if os.path.exists(f)]

    if not valid_files:
        return

    dfs = []

    for f in valid_files:
        df = pd.read_csv(f, dtype={"Symbol": str})
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return

    final_oof_df = pd.concat(dfs, ignore_index=True)

    final_oof_df["Symbol"] = final_oof_df["Symbol"].astype(str).str.zfill(6)
    final_oof_df["Date_Temp"] = pd.to_datetime(final_oof_df["Date"])

    final_oof_df = (
        final_oof_df.sort_values(["Date_Temp", "Symbol"])
        .drop(columns=["Date_Temp"])
        .reset_index(drop=True)
    )

    final_oof_df.to_csv(output_csv, index=False, encoding="utf_8_sig")


def generate_oof(dataset: str = DEFAULT_DATASET) -> tuple[str, str, str]:
    configure_paths(dataset)
    ensure_dirs()
    check_required_files()

    best_config = {
        "run_name": f"compact_Bstyle_OOF_{dataset}",
        "save_name": f"temp_compact_Bstyle_oof_{dataset}",
        "r": 16,
        "lora_alpha": 32,
        "learning_rate": 5e-5,
        "epochs": 16,
        "batch_size": 1,
        "grad_acc_steps": 8,
        "early_stopping_patience": 5,
    }

    oof_prediction_files = []
    for i in range(1, len(TIME_BLOCKS)):
        current_block_idx = i + 1
        train_end_date = TIME_BLOCKS[i - 1][1]
        infer_start_date = TIME_BLOCKS[i][0]
        infer_end_date = TIME_BLOCKS[i][1]

        (
            curr_train_jsonl,
            curr_eval_jsonl,
            curr_infer_csv,
            train_rows,
            infer_rows,
        ) = prepare_temporal_data(
            raw_csv_path=RAW_TRAIN_CSV,
            raw_jsonl_path=RAW_TRAIN_JSONL,
            train_end_date=train_end_date,
            infer_start_date=infer_start_date,
            infer_end_date=infer_end_date,
            current_block=current_block_idx,
        )

        if train_rows == 0 or infer_rows == 0:
            continue

        block_config = best_config.copy()
        block_config["run_name"] = (
            f"compact_Bstyle_oof_{dataset}_block_{current_block_idx}"
        )
        block_config["save_name"] = (
            f"temp_compact_Bstyle_lora_{dataset}_block_{current_block_idx}"
        )

        temp_lora_path = run_training(
            config=block_config,
            train_data_path=curr_train_jsonl,
            eval_data_path=curr_eval_jsonl,
        )
        block_output_csv = OOF_PRED_CSV_TPL.format(current_block_idx)
        ok = run_inference(
            lora_path=temp_lora_path,
            input_csv=curr_infer_csv,
            output_csv=block_output_csv,
        )
        if ok:
            oof_prediction_files.append(block_output_csv)

        for temp_file in (curr_train_jsonl, curr_eval_jsonl, curr_infer_csv):
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if DELETE_TEMP_LORA_AFTER_INFER and os.path.exists(temp_lora_path):
            shutil.rmtree(temp_lora_path, ignore_errors=True)

    merge_oof_predictions(oof_prediction_files, FINAL_OOF_SAFE_TRAIN_CSV)

    if RUN_VAL_INFERENCE and os.path.exists(BEST_FULL_LORA_PATH):
        run_inference(BEST_FULL_LORA_PATH, RAW_VAL_CSV, FINAL_VAL_PRED_CSV)
    if RUN_TEST_INFERENCE and os.path.exists(BEST_FULL_LORA_PATH):
        run_inference(BEST_FULL_LORA_PATH, RAW_TEST_CSV, FINAL_TEST_PRED_CSV)

    required_outputs = [FINAL_OOF_SAFE_TRAIN_CSV]
    if RUN_VAL_INFERENCE:
        required_outputs.append(FINAL_VAL_PRED_CSV)
    if RUN_TEST_INFERENCE:
        required_outputs.append(FINAL_TEST_PRED_CSV)
    missing = [path for path in required_outputs if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"OOF generation did not create required outputs: {missing}")

    print(
        f"Saved train OOF and val/test Student predictions for {dataset} "
        f"to {DATA_DIR}"
    )
    return FINAL_OOF_SAFE_TRAIN_CSV, FINAL_VAL_PRED_CSV, FINAL_TEST_PRED_CSV


def main() -> None:
    args = parse_args()
    generate_oof(args.dataset)


if __name__ == "__main__":
    main()
