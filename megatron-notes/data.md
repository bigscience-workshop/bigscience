# Data processing

## pre-processing

Instructions are here:
https://github.com/NVIDIA/Megatron-LM/#data-preprocessing


## Merging multiple pre-processed indexed datasets

TODO: Need to write a script that merges existing indices - this is needed for situation when we either:
1. don't have enough SLURM hours to complete pre-processing when the input is huge
2. already have sub-datasets preprocessed and we want to re-use those, rather than pre-processing everything from scratch

Direction using `merge_file_`
https://github.com/NVIDIA/Megatron-LM/blob/90e0a0dd08159e1c95f4f9d99bb8687f327d36c3/megatron/data/indexed_dataset.py#L294
It looks like you have to create a new builder, and merge all already processed documents to it.
