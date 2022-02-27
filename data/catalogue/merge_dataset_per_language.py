import argparse
import json
from collections import defaultdict

import regex as re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-ratios-path",
        type=str,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'ratio': float}",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"]
    )
    parser.add_argument(
        "--meg-ds-dataset-prefix",
        type=str,
        required=True,
        help="We add `lang` to that prefix in order to designate the path for a languages specific dataset."
    )
    parser.add_argument(
        "--output-ratio-file",
        type=str,
        required=True,
        help="path to output the language ratio file",
    )
    return parser.parse_args()

TOKEN_RANGES={
    "train": "0:0.949",
    "valid": "0.949:0.999",
    "test": "0.999:1.0"
}

def main():
    args = get_args()

    # load training datasets
    with open(args.dataset_ratios_path, "r") as fi:
        ds_ratios = json.load(fi)

    # get all individual languages
    r = re.compile(r"^.*bigscience-catalogue-lm-data/lm_([^_]+)_.*$")
    datasets_per_language = defaultdict(lambda: [])
    for ds_ratio in ds_ratios:
        candidate_lang = r.match(ds_ratio["dataset_path"]).group(1)
        if candidate_lang == "hi":
            ds_ratio["lang"] = "indic-hi"
        else:
            ds_ratio["lang"] = candidate_lang
        datasets_per_language[ds_ratio["lang"]].append(ds_ratio)

    # save ratio result into a file (in json format, you can use `load_ratios_meg_ds_format` for get the meg_ds format)
    language_ds_ratios = [
        {
            "ratio": sum([elt["ratio"] for elt in datasets]),
            "dataset_path": args.meg_ds_dataset_prefix.format(lang=lang),
            # Additional field to store in case we want to know what's in there.
            "original_datasets": [
                dataset["dataset_path"]
                for dataset in datasets
            ]
        }
        for lang, datasets in datasets_per_language.items()
    ]

    with open(args.output_ratio_file, "w") as fi:
        json.dump(language_ds_ratios, fi, indent=2)

if __name__ == "__main__":
    main()
