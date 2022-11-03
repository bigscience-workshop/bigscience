import argparse
import json
import re
from pathlib import Path
from re import Pattern
from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path, help="Path to the list of results")
    parser.add_argument("--concatenate-output-file", required=True, type=Path, help="Path to store the final output file")
    return parser.parse_args()

MODEL = "tr11b-1b3-ml-bsevalharness-results_lm-eval_global_step340500"
# MODEL = "global_step95000"
RESULTS_REGEX = re.compile(rf"(eai|bs)_results_lm-eval_{MODEL}_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_backup\.json")
RESULTS_REGEX = re.compile(rf"{MODEL}_*.json")
#tr11b-1b3-ml-bsevalharness-results_lm-eval_global_step340500_2022-07-14-10-03-25.json
def get_all_files_that_match_results_in_folder(root_folder: Path) -> List[Path]:
    json_files = []
    for folder in root_folder.iterdir():
        if folder.is_dir():
            json_files += get_all_files_that_match_results_in_folder(folder)
        else:
            # it's actually a file
            file = folder

            #match = RESULTS_REGEX.match(file.name)

            if not str(file.name).endswith("json"):
                continue
            else:
                json_files.append(file)
    return json_files

def sort_dict(dictionary: Dict) -> Dict:
    results = {}

    for key, value in sorted(dictionary.items()):
        new_value = value

        if isinstance(value, dict):
            new_value = sort_dict(new_value)
        elif isinstance(value, list):
            new_value = sorted(value)

        results[key] = new_value

    return results

def main():
    args = get_args()

    # Get all json files
    json_files = get_all_files_that_match_results_in_folder(args.results_dir)
    print("GOT", json_files)
    # Merge all json files
    final_result = {
        "results": {},
        "versions": {}
    }
    for file in json_files:
        with open(file, "r") as fi:
            task_result = json.load(fi)

        #match = RESULTS_REGEX.match(file.name)
        #assert match is not None
        prefix = "bs" if "bs" in file.name else "eai"#match.group(1)
        datetime_string = file.name[file.name.index("global_step340500_") + len("global_step340500_"):].replace(".json", "")#match.group(2)

        if prefix == "eai":
            results_key = "results"
        elif prefix == "bs":
            results_key = "table_results"
        else:
            raise ValueError(f"Unsupported key: {prefix}")

        for key, value in task_result[results_key].items():
            if key not in final_result["results"]:
                final_result["results"][key] = {
                    datetime_string: value
                }
            #else:
            #    assert datetime_string not in final_result["results"][key]
            #    final_result["results"][key][datetime_string] = value

        for key, value in task_result["versions"].items():
            final_result["versions"][key] = value

    # We sort dict, better for serialization
    print(final_result)
    final_result = sort_dict(final_result)

    # Save result
    with open(args.concatenate_output_file, "w") as fo:
        json.dump(final_result, fo, indent=2)

    pass

if __name__ == "__main__":
    main()

