#!/usr/bin/env python
#
# generate jsonl version of dataset that can be fed to megatron-lm pre-processor
#
# see various notes in the scripts for different options
#
# full dataset:
# ./oscar_to_jsonl.py
# cat oscar-[0-4].jsonl > oscar.jsonl
#
# small dataset (0.1%):
# ./oscar_to_jsonl.py -s
# cat oscar-[0-4].jsonl > oscar.jsonl

import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count, Process, Queue

from datasets import concatenate_datasets, load_dataset, ReadInstruction
from transformers import GPT2TokenizerFast

import datasets
print(f"Using datasets=={datasets.__version__}")

DATASET_NAME = "oscar"
CONTEXT_SIZE = 1024

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# used by map
def filter_short_documents(batch_documents):
    results = {"id": [], "text": []}

    tokenized_documents = tokenizer(batch_documents["text"], return_length=True)

    for i in range(len(batch_documents["id"])):
        if tokenized_documents.length[i] >= CONTEXT_SIZE:
            results["id"].append(batch_documents["id"][i])
            results["text"].append(batch_documents["text"][i])
    return results

# used by filter
def is_big(batch_documents):
    tokenized_documents = tokenizer(batch_documents["text"], return_length=True)
    return [length >= CONTEXT_SIZE for length in tokenized_documents.length]

parser = ArgumentParser()
parser.add_argument('-s', '--subset', action='store_true', help='Process and save a subset (0.1%) of data')
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Once this part of the process runs it gets cached, so on subsequent runs it'll be much faster

split = ReadInstruction("train", to=0.1 if args.subset else 100, unit="%")

### Build/Load Datasets

# Once this part of the process completes it gets cached, so on subsequent runs it'll be much faster

language_subsets = (
    # "unshuffled_deduplicated_ar",
    # "unshuffled_deduplicated_sw",
    # "unshuffled_deduplicated_zh",
    "unshuffled_deduplicated_en",
    # "unshuffled_deduplicated_fr",
    # "unshuffled_deduplicated_pt",
    # "unshuffled_deduplicated_es",
)


datasets = []

for language_subset in language_subsets:
    dataset = load_dataset(DATASET_NAME, language_subset, split=split, keep_in_memory=False, cache_dir='cache')
    datasets.append(dataset)

### Filter large records

# Once this part of the process completes it gets cached, so on subsequent runs it'll be much faster

concat_dataset = concatenate_datasets(datasets)
print(f"Filtering {len(concat_dataset)} examples")

# version 1 using map:
# takes about 8-10h
filtered_dataset = concat_dataset.map(filter_short_documents, batched=True, batch_size=256, num_proc=cpu_count())

# version 2 using the experimental 'optimize-filter' branch of datasets
# this should be faster as it manipulates the indices - less disc space used as well
# didn't run fully, but based on ETAs didn't suggest to finish any faster than version 1
#filtered_dataset = concat_dataset.filter(is_big, load_from_cache_file=True, batched=True, num_proc=cpu_count())

print(f"Before filtering: {len(concat_dataset)} examples, after filtering: {len(filtered_dataset)} examples")

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
SHARDS = 5
def process_shard(idx):
    print(f"Sharding {idx}")
    ds_shard = filtered_dataset.shard(SHARDS, idx, contiguous=True)
    # shuffle will make things much much slower
    #ds_shard = ds_shard.shuffle() # remove contiguous=True above if shuffling
    print(f"Saving {DATASET_NAME}-{idx}.jsonl")
    ds_shard.to_json(f"{DATASET_NAME}-{idx}.jsonl", orient="records", lines=True, force_ascii=False)

queue = Queue()
processes = [Process(target=process_shard, args=(idx,)) for idx in range(SHARDS)]
for p in processes:
    p.start()

for p in processes:
    p.join()
