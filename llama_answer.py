
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


if __name__ == "__main__":
    model = AutoPeftModelForCausalLM.from_pretrained('/mnt/Data/ml/training/finetune/', device_map="auto", torch_dtype=torch.bfloat16)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained('./llama2-transformed/', use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer('Answer the following question as I am 5 years old and do not repeat yourself while ansering: You are a attacker trying to infiltrate a cloud network. What is the Provider type to enter to OIDC?', return_tensors='pt').to('cpu')

    # generate 40 new tokens
    greedy_output = model.generate(**model_inputs, max_new_tokens=400)[0]
    print(tokenizer.decode(greedy_output, skip_special_tokens=True))
    