import json
import os
from pathlib import Path

def main():
    """Write to """
    data_path = Path(os.environ["six_ALL_CCFRSCRATCH"]) / "datasets-custom" / "mc4" / "mc4_preprocessing"
    output_path = Path(os.environ["six_ALL_CCFRSCRATCH"]) / "checkpoints" / "tr5-1B3-multilingual" / "dataset_probabilities.txt"

    probabilies_path = data_path / "sample_iterator_probs" / "iterator_selection_prob.0.3.train.json"

    with open(probabilies_path, "r") as fprob:
        probabilities = json.load(fprob)

    # Format probabilities dictionary to store path in key and probability as value
    probabilities = {
        data_path / key.replace("dumped/mc4_processed_data/", ""): value[0] for key, value in probabilities.items()
    }

    with open(output_path, "w") as fout:
        fout.write(" ".join([f"{prob} {path}" for path, prob in probabilities.items()]))
    pass

if __name__ == "__main__":
    main()
