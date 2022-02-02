#!/bin/bash
# the $1 argument can be "equal" or "alpha" to choose the corresponding tokenizer
# no en in this script; we've mostly processed it on GCP
for language in fr es zh hi ur bn id ca ar pt vi eu
do
    sbatch $ALL_CCFRWORK/code/bigscience/data/oscar-multilingual/oscar-jsonl-to-meg $language $1
done
