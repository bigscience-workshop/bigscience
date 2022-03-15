import argparse
import json


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
        "--output-meg-ds-ratio-file",
        type=str,
        required=True,
        help="path to output the language ratio file",
    )
    return parser.parse_args()

TOKEN_RANGES={
    "train": "0:0.950",
    "valid": "0.950:1.0",
}

def main():
    args = get_args()

    token_range = TOKEN_RANGES[args.split]

    with open(args.dataset_ratios_path, "r") as fi:
        ds_ratios = json.load(fi)

    main_dataset = [f"{ds_ratio['ratio']} {token_range} {ds_ratio['dataset_path']}" for ds_ratio in ds_ratios]
    if args.split == "train":
        final_string = f"\"{args.split}: " + ", ".join(main_dataset) + "\"\n"
    elif args.split == "valid":
        main_dataset_string = f"\"{args.split}_all: " + ", ".join(main_dataset) + "\""
        additional_datasets = [f"\"valid_{ds_ratio['dataset_path'].split('/')[-2]}: 1 {token_range} {ds_ratio['dataset_path']}\"" for ds_ratio in ds_ratios]
        final_string = main_dataset_string + " " + " ".join(additional_datasets)
    else:
        raise ValueError(f"unknown split string {args.split}")


    # TODO: you can add some extra dataset names for validation/test
    with open(args.output_meg_ds_ratio_file, "w") as fi:
        fi.write(final_string)

if __name__ == "__main__":
    main()
