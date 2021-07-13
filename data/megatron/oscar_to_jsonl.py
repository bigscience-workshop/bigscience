#!/usr/bin/env python
# generate jsonl version of dataset that can be fed to megatron-lm preprocessor
#
# full dataset
# ./oscar_to_jsonl.py
#
# small dataset
# ./oscar_to_jsonl.py -s
import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count

from datasets import concatenate_datasets, load_dataset, ReadInstruction
from transformers import GPT2TokenizerFast

DATASET_NAME = "oscar"
CONTEXT_SIZE = 2048

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def filter_short_documents(batch_documents):
    results = {"id": [], "text": []}

    tokenized_documents = tokenizer(batch_documents["text"], return_length=True)

    for i in range(len(batch_documents["id"])):
        if tokenized_documents.length[i] >= CONTEXT_SIZE:
            results["id"].append(batch_documents["id"][i])
            results["text"].append(batch_documents["text"][i])
    return results


parser = ArgumentParser()
parser.add_argument('-s', '--subset', action='store_true', help='Process and save a subset (0.1%) of data')
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

split = ReadInstruction("train", to=0.1 if args.subset else 100, unit="%")

language_subsets = (
    "unshuffled_deduplicated_ar",
    "unshuffled_deduplicated_sw",
    # "unshuffled_deduplicated_zh",
    # "unshuffled_deduplicated_en",
    "unshuffled_deduplicated_fr",
    "unshuffled_deduplicated_pt",
    "unshuffled_deduplicated_es",
)

datasets = []

for language_subset in language_subsets:
    dataset = load_dataset(DATASET_NAME, language_subset, split=split, keep_in_memory=False, cache_dir='cache')
    datasets.append(dataset)

concat_dataset = concatenate_datasets(datasets)
filtered_dataset = concat_dataset.map(filter_short_documents, batched=True, batch_size=256, num_proc=cpu_count())

print(f"Before filtering: {len(concat_dataset)} examples, after filtering: {len(filtered_dataset)} examples")

shuffled_dataset = filtered_dataset.shuffle()
shuffled_dataset.to_json(f"{DATASET_NAME}.jsonl", orient="records", lines=True, force_ascii=False)
