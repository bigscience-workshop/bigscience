#!/usr/bin/env python
#
# generate jsonl version of dataset that can be fed to megatron-lm pre-processor
#
# see various notes in the scripts for different options
#
# full dataset:
# ./oscar-multilingual-to-jsonl.py
# cat oscar-[0-4].jsonl > oscar.jsonl
#
# small dataset (0.1%):
# ./oscar-multilingual-to-jsonl.py -s
# cat oscar-[0-4].jsonl > oscar.jsonl

import logging
from argparse import ArgumentParser
from multiprocessing import Process, Queue

from datasets import load_dataset, ReadInstruction

import datasets

print(f"Using datasets=={datasets.__version__}")

DATASET_NAME = "oscar"

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

parser = ArgumentParser()
parser.add_argument('-s', '--subset', action='store_true', help='Process and save a subset (0.1%) of data')
args = parser.parse_args()

# Once this part of the process runs it gets cached, so on subsequent runs it'll be much faster

split = ReadInstruction("train", to=0.1 if args.subset else 100, unit="%")

### Build/Load Datasets

# Once this part of the process completes it gets cached, so on subsequent runs it'll be much faster

language_subsets = {
    # "unshuffled_deduplicated_hi",
    "unshuffled_deduplicated_ur",
    # "unshuffled_deduplicated_bn",
    "unshuffled_deduplicated_id",
    # "unshuffled_deduplicated_am",
    # "unshuffled_deduplicated_ca",
    # "unshuffled_deduplicated_ru",
    # "unshuffled_deduplicated_ar",
    # "unshuffled_deduplicated_sw",
    # "unshuffled_deduplicated_zh",
    # "unshuffled_deduplicated_en",
    # "unshuffled_deduplicated_fr",
    # "unshuffled_deduplicated_pt",
    # "unshuffled_deduplicated_es",
    # "unshuffled_deduplicated_ja",
}
sharded_languages = {
    "unshuffled_deduplicated_en",
    "unshuffled_deduplicated_ru",
    "unshuffled_deduplicated_de",
    "unshuffled_deduplicated_es",
    "unshuffled_deduplicated_fr",
    "unshuffled_deduplicated_ja",
    "unshuffled_deduplicated_zh",
}

### Save jsonl

# important: shuffling makes the process 5-7 times slower! best to shuffle the end jsonl file using
# https://github.com/alexandres/terashuf (should take ~1h to shuffle 900GB file with 70M records
# using 150GB RAM)

# version 1: one writer - quite slow
#shuffled_dataset = filtered_dataset.shuffle()
#shuffled_dataset = filtered_dataset
#shuffled_dataset.to_json(f"{DATASET_NAME}.jsonl", orient="records", lines=True, force_ascii=False)

# version 2: multiple parallel sharded writes
# much faster, but will require concatenation at the end
# 10 shards proved to much for the instance and 3 processed were killed, 5 worked well
# took about 1.5h per shard

N_SHARDS = 5
def process_shard(dataset, n_shards, idx, language_subset):
    if n_shards > 1:
        print(f"Sharding {idx}")
        ds_shard = dataset.shard(n_shards, idx, contiguous=True)
        # shuffle will make things much much slower
        #ds_shard = ds_shard.shuffle() # remove contiguous=True above if shuffling
    else:
        ds_shard = dataset
    print(f"Saving {DATASET_NAME}-{language_subset}-{idx}.jsonl")
    export_filename = f"{DATASET_NAME}-{language_subset}-{idx}.jsonl" if n_shards > 1 else \
        f"{DATASET_NAME}-{language_subset}.jsonl"
    ds_shard.to_json(export_filename, orient="records", lines=True, force_ascii=False)

for language_subset in language_subsets:
    n_shards = N_SHARDS if language_subset in sharded_languages else 1
    dataset = load_dataset(DATASET_NAME, language_subset, split=split, keep_in_memory=False, cache_dir='cache', ignore_verifications=True)
    queue = Queue()
    processes = [Process(target=process_shard, args=(dataset, n_shards, idx, language_subset,)) for idx in range(n_shards)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
