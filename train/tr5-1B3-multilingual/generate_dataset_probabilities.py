import json
import os
from pathlib import Path

def removeprefix(string, prefix):
    if string.startswith(prefix):
        string = string[len(prefix):]
    return string

def removesuffix(string, suffix):
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string

def convert_path(original_path, data_path):
    prefix_to_remove = "dumped/mc4_processed_data/"
    suffix_to_remove = ".bin"

    return data_path / removesuffix(removeprefix(original_path, prefix_to_remove), suffix_to_remove)

def main():
    """Write to """
    data_path = Path(os.environ["six_ALL_CCFRSCRATCH"]) / "datasets-custom" / "mc4" / "mc4_preprocessing"
    output_path = Path(os.environ["six_ALL_CCFRSCRATCH"]) / "checkpoints" / "tr5-1B3-multilingual" / "dataset_probabilities.txt"

    probabilies_path = data_path / "sample_iterator_probs" / "iterator_selection_prob.0.3.train.json"

    with open(probabilies_path, "r") as fprob:
        probabilities = json.load(fprob)

    # Format probabilities dictionary to store path in key and probability as value
    probabilities = {
        convert_path(key, data_path): value[0] for key, value in probabilities.items()
    }

    with open(output_path, "w") as fout:
        fout.write(" ".join([f"{prob} {path}" for path, prob in probabilities.items()]))
    pass

if __name__ == "__main__":
    main()
