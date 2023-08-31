# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Union

import fire
import torch
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    cleanup,
    clear_gpu_cache,
    get_parameter_dtypes,
    print_model_size,
    get_policies  
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model
import configs

import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import argparse
import bitsandbytes as bnb
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments


def create_bnb_config():
    bits = 4
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16 ,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear8bitLt #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def load_model(model_name, bnb_config):
    print('Load model: ' + model_name)
    n_gpus = torch.cuda.device_count()
    max_memory = f'{22888}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train(model, tokenizer, dataset, eval_dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            gradient_accumulation_steps=4,
            num_train_epochs=18,
            warmup_steps=2,
            learning_rate=2e-4,
            logging_steps=1,
            fp16=True,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

def main(**kwargs):
    print('Merging the configs')
    update_config((train_config, fsdp_config), **kwargs)
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(train_config.model_name, bnb_config)
    dataset_config = generate_dataset_config(train_config, kwargs)
    
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    
    print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print(f"--> test Set Length = {len(dataset_test)}")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)



    train(model, tokenizer, dataset_train, dataset_test, train_config.output_dir)
    
    #model should be deleted in training
    model = AutoPeftModelForCausalLM.from_pretrained(train_config.output_dir, 
                                                     device_map="auto", 
                                                     load_in_4bit = True,
                                                     torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = "results/llama2/final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)

if __name__ == "__main__":
    fire.Fire(main)
