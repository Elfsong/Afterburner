import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM


dataset = load_dataset("trl-lib/tldr", split="train")
model_name = 'Qwen/Qwen2.5-Coder-3B'

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(
    output_dir=f"{model_name}-GRPO", 
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,
    report_to="wandb",
    bf16=True,
    use_vllm=True
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()