import argparse
import json
import re
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path, help="Path to the list of results")
    parser.add_argument("--concatenate-output-file", required=True, type=Path, help="Path to store the final output file")
    return parser.parse_args()

def main():
    args = get_args()

    # Get all json files
    json_files = []
    for folder in args.results_dir.iterdir():
        if folder.is_file():
            continue
        for file in folder.iterdir():
            if file.is_dir():
                continue
            match = re.match(
                r"(?:eai|bs)_results_lm-eval_opt-175b-meg-ds_(?:\d{4})-(?:\d{2})-(?:\d{2})-(?:\d{2})-(?:\d{2})-(?:\d{2})\.json",
                file.name,
            )

            if match is None:
                continue
            else:
                # TODO @thomasw21 some folder can have multiple results we should take the latest
                json_files.append(file)
                break

    # Merge all json files
    final_result = {
        "results": {},
        "versions": {}
    }
    for file in json_files:
        with open(file, "r") as fi:
            task_result = json.load(fi)

        for key, value in task_result["results"].items():
            final_result["results"][key] = value

        for key, value in task_result["versions"].items():
            final_result["versions"][key] = value

    # Save result
    with open(args.concatenate_output_file, "w") as fo:
        json.dump(final_result, fo)

    pass

if __name__ == "__main__":
    main()
