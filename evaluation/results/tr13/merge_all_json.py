import argparse
import json
from pathlib import Path
from typing import Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--save-output-json", type=str)
    return parser.parse_args()

def find_all_json(root_dir: Path):
    if root_dir.is_file():
        if root_dir.name.endswith(".json"):
            return [root_dir]
        else:
            return []

    all_jsons = []
    for path in root_dir.iterdir():
        all_jsons += find_all_json(path)
    return all_jsons

def sort_dict(dictionary: Dict) -> Dict:
    results = {}

    for key, value in sorted(dictionary.items(), key=lambda item: item[0]):
        new_value = value

        if isinstance(value, dict):
            new_value = sort_dict(new_value)
        elif isinstance(value, list):
            new_value = sorted(value)

        results[key] = new_value

    return results

def main():
    args = get_args()

    # find all json file in directory
    root_dir: Path = args.directory
    all_jsons = find_all_json(root_dir)

    # merge
    results = {}
    for json_file in all_jsons:
        with open(json_file, "r") as fi:
            data = json.load(fi)

        key = f"{data['dataset_name']}_{data['dataset_config_name']}"
        if key in results:
            assert data["template_name"] not in results
            results[key][data["template_name"]] = data
        else:
            results[key] = {
                data["template_name"]: data
            }

    # sort
    sorted_results = sort_dict(results)

    # write
    with open(args.save_output_json, "w") as fo:
        json.dump(sorted_results, fo)


if __name__ == "__main__":
    main()
