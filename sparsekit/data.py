from typing import Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

import numpy as np
import random


class RandomTokens(Dataset):
    def __init__(self, tokenizer, seq_len, size=1_000, seed=None, **kwargs):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.size = size

        special_ids = set(
            [
                getattr(tokenizer, a)
                for a in [
                    "eos_token_id",
                    "pad_token_id",
                    "unk_token_id",
                ]
                if getattr(tokenizer, a, None) is not None
            ]
        )

        self.allowed_ids = torch.tensor(
            [i for i in range(tokenizer.vocab_size) if i not in special_ids]
        ).long()

        self._rand = torch.Generator()
        if seed is not None:
            self._rand.manual_seed(int(seed))

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        idxs = torch.randint(
            low=0,
            high=len(self.allowed_ids),
            size=(self.seq_len,),
            generator=self._rand,
            dtype=torch.long,
        )
        input_ids = self.allowed_ids[idxs]
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return input_ids, attention_mask


def get_llm_dataset(hub_path, split="train", **kwargs):
    path, name = hub_path.split("/")
    return load_dataset(path, name, split=split, **kwargs)


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(num_samples, seq_len, tokenizer, seed=42, split="train"):
    # Load train and validation datasets
    allen_c4_ds = load_dataset(
        "allenai/c4",
        "en",
        # data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz",
        },
        verification_mode="no_checks",
        split=split,
    )

    trainloader = []
    long_samples = []
    random.seed(seed)

    permutation = list(range(len(allen_c4_ds)))
    random.shuffle(permutation)
    for t_idx in tqdm(permutation):
        sample_text = allen_c4_ds[t_idx]["text"]
        sample_tokens = tokenizer(sample_text, return_tensors="pt")
        if sample_tokens.input_ids.shape[1] >= seq_len:
            long_samples.append(sample_tokens)
        if len(long_samples) >= num_samples:
            break

    if len(long_samples) < num_samples:
        raise ValueError("Not enough long samples found.")

    print("Clipping samples to length", seq_len)

    for tokens in tqdm(long_samples):
        start = random.randint(0, tokens.input_ids.shape[1] - seq_len)
        inputs = tokens.input_ids[:, start : start + seq_len]
        targets = inputs.clone()
        targets[:, :-1] = -100
        trainloader.append((inputs, targets))

    return trainloader


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)


def sharded_dataset(
    dataset_name,
    dataset_partition,
    num_shards,
    shard_id,
    streaming=True,
    split="train",
):
    pass
