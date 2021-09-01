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
import os

import datasets

print(f"Using datasets=={datasets.__version__}")

DATASET_NAME = "oscar"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

### Build/Load Datasets

# Once this part of the process completes it gets cached, so on subsequent runs it'll be much faster

language_subsets = {
    "unshuffled_deduplicated_ar",
    "unshuffled_deduplicated_sw",
    "unshuffled_deduplicated_zh",
    # "unshuffled_deduplicated_en",
    "unshuffled_deduplicated_fr",
    "unshuffled_deduplicated_pt",
    "unshuffled_deduplicated_es",
    "unshuffled_deduplicated_ja",
    "unshuffled_deduplicated_ru",
    "unshuffled_deduplicated_hi",
    "unshuffled_deduplicated_ur",
    "unshuffled_deduplicated_bn",
    "unshuffled_deduplicated_id",
    "unshuffled_deduplicated_am",
    "unshuffled_deduplicated_ca",
}

for language_subset in language_subsets:
    builder = datasets.load_dataset_builder(DATASET_NAME, language_subset, cache_dir='cache')
    if not os.path.isdir(builder.cache_dir):
        builder.download_and_prepare(ignore_verifications=True)
