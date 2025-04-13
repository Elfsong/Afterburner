import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, HfFolder, Repository

def merge_and_save(base_model_path, lora_model_path, merged_output_dir, hf_repo_id, commit_message):
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    print("🔗 Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    print("✅ Merging LoRA into base model...")
    model = model.merge_and_unload()
    print(f"💾 Saving merged model to {merged_output_dir}...")
    model.save_pretrained(merged_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_output_dir)
    print("✅ Model and tokenizer saved.")

def upload_to_hub(merged_output_dir, hf_repo_id, commit_message):
    print("🌠 Uploading to Hugging Face Hub...")
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("❌ You are not logged in. Run `huggingface-cli login` first.")

    api = HfApi()

    # Create repo if not exists
    if not hf_repo_id.startswith("https://huggingface.co/"):
        full_url = f"https://huggingface.co/{hf_repo_id}"
    else:
        full_url = hf_repo_id

    try:
        print(f"📡 Creating or accessing repo: {hf_repo_id}")
        api.create_repo(repo_id=hf_repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"⛔️ Warning: Could not create repo. Might already exist: {e}")

    print("🚀 Uploading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(merged_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(merged_output_dir)

    model.push_to_hub(hf_repo_id, commit_message=commit_message)
    tokenizer.push_to_hub(hf_repo_id, commit_message=commit_message)

    print(f"✅ Upload completed. View at: {full_url}")

if __name__ == "__main__":    
    # ==== CONFIGURATION ====
    base_model_path = "Qwen/Qwen2.5-3B-Instruct"
    lora_model_path = "/home/mingzhe/Projects/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/qwen_3b_dpo_batch_8_lr_4e-5/checkpoint-800"
    merged_output_dir = "./merged-qwen-3b-dpo-batch-8-lr-4e-5"
    hf_repo_id = "Elfsong/Qwen2.5-3B-DPO-Batch-8-LR-4e-5"
    commit_message = "Upload merged Qwen2.5-3B-DPO-Batch-8-LR-4e-5"
    # ========================
    merge_and_save(base_model_path, lora_model_path, merged_output_dir, hf_repo_id, commit_message)
    upload_to_hub(merged_output_dir, hf_repo_id, commit_message)
