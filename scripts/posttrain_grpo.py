import os
import subprocess
import sys

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

# --- Configuration ---
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "output"

# Eval settings
RUN_EVAL = True
EVAL_BENCHMARKS = [
    {"data_id": "zwhe99/MATH", "split": "math500"},
]
EVAL_TENSOR_PARALLEL_SIZE = 1
EVAL_TEMPERATURE = 0.6
EVAL_N = 1  # number of samples per problem
EVAL_MAX_MODEL_LEN = 32768

# --- Training ---
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    report_to="wandb",
    learning_rate=1e-6,
    beta=0.001,
    max_completion_length=4096,
    num_generations=4,
    temperature=0.6,
    max_steps=1800,
    max_grad_norm=5.0,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    use_vllm=False, # restricted to 1 task on 1 GPU
)

trainer = GRPOTrainer(
    model=BASE_MODEL,
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()

# Save the final model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

# --- Evaluation ---
if RUN_EVAL:
    trained_model_path = os.path.join(OUTPUT_DIR, "final_model")
    eval_script = os.path.join(os.path.dirname(__file__), "uni_eval.py")

    for benchmark in EVAL_BENCHMARKS:
        data_id = benchmark["data_id"]
        split = benchmark.get("split")
        eval_output_dir = os.path.join(OUTPUT_DIR, "eval", data_id.replace("/", "_"))
        if split:
            eval_output_dir = os.path.join(eval_output_dir, split)

        cmd = [
            sys.executable, eval_script,
            "--base_model", trained_model_path,
            "--output_dir", eval_output_dir,
            "--data_id", data_id,
            "--tensor_parallel_size", str(EVAL_TENSOR_PARALLEL_SIZE),
            "--temperature", str(EVAL_TEMPERATURE),
            "--n", str(EVAL_N),
            "--max_model_len", str(EVAL_MAX_MODEL_LEN),
            "--bf16",
        ]
        if split:
            cmd.extend(["--split", split])

        print(f"\n{'='*60}")
        print(f"Running evaluation: {data_id} (split={split})")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)
