#!/usr/bin/env python

# generate jsonl version of dataset that can be fed to megatron-lm preprocessor
#
# full dataset
# ./openwebtext-to-jsonl.py
#
# 10k small dataset
# ./openwebtext-to-jsonl.py -10k

import sys
from datasets import load_dataset

if "-10k" in sys.argv:
    dataset_name = "stas/openwebtext-10k"
else:
    dataset_name = "openwebtext"

name = dataset_name.split('/')[-1]
ds = load_dataset(dataset_name, split='train')
ds.to_json(f"{name}.jsonl", orient="records", lines=True)

# subset to jsonlines
# n_samples = 10000
# ds = load_dataset(dataset_name, split='train')
# ds_small = ds.select(range(n_samples))
# path = f"{dataset_name}-{n_samples}.jsonl"
# ds.to_json(path, orient="records", lines=True)

