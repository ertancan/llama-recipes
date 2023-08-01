# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

class CloudF6sDataset(Dataset):
    # TODO: modify the prompt
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{prompt}\n\n### Response:"
    )
    def __init__(self, dataset_config, tokenizer, split="train", max_words = 4096):
        if split == 'train':
            filename = 'training_data/training/data.json'
        else:
            filename = 'training_data/validation/data.json'
        self.max_words = max_words
        self.examples = json.load(open(filename))

        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        data_row = self.examples[index]

        prompt = self.prompt.format_map(data_row)
        
        example = prompt + data_row["completion"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
