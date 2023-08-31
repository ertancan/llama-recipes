
import torch
from peft import AutoPeftModelForCausalLM

if __name__ == "__main__":
    model = AutoPeftModelForCausalLM.from_pretrained('/mnt/Data/ml/training/finetune/', device_map="auto", torch_dtype=torch.bfloat16)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained('/mnt/Data/ml/training/finetune_merged/', safe_serialization=False)