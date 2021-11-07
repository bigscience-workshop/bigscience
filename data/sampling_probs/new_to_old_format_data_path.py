import argparse
import os.path
from glob import glob
import json

SPLIT = [0, 0.949, 0.999, 1.0]


def finalize_dataset_string(dataset_string):
    # remove trailing comma in case
    # surround with quotes
    if dataset_string.endswith(","):
        dataset_string = dataset_string[:-1]
    return '"' + dataset_string + '"'


def get_longest_prefix_and_suffix(file1, file2):
    # we're assuming all filepaths have the same format
    prefix = max([i for i in range(len(file1)) if file2.startswith(file1[:i])])
    suffix = min([i for i in range(len(file1)-1, -1, -1) if file2.endswith(file1[i:])])
    return prefix, suffix



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files-dir', type=str, required=True,
                        help='Path to the data folder')
    args = parser.parse_args()

    for filename in glob(f'{args.input_files_dir}/*.json'):
        sampling_probs = json.load(open(filename))
        # assuming alpha is of the form 0.x, this could break
        alpha = filename[-8:-5]
        # we remove the .bin at the end of the filename
        file_weights = [(k[:-4], v[0]) for k, v in sampling_probs.items()]

        prefix, suffix = get_longest_prefix_and_suffix(file_weights[0][0], file_weights[1][0])

        train_split_string = f"{SPLIT[0]}:{SPLIT[1]}"
        valid_split_string = f"{SPLIT[1]}:{SPLIT[2]}"
        test_split_string = f"{SPLIT[2]}:{SPLIT[3]}"

        train_string = f"train:"
        for file, weight in file_weights:
            train_string += f" {weight} {train_split_string} {file},"
        train_string = finalize_dataset_string(train_string)
        with open(os.path.join(args.input_files_dir, f"train_data_string.{alpha}.txt"), "w") as f:
            f.write(train_string)

        valid_strings = ["all_valid:"]
        for file, weight in file_weights:
            valid_strings[0] += f" {weight} {valid_split_string} {file},"
            language_code = file[prefix:suffix]
            valid_strings.append(f"valid_{language_code}: 1 {valid_split_string} {file}")
        valid_string = " ".join([finalize_dataset_string(valid_string) for valid_string in valid_strings])
        with open(os.path.join(args.input_files_dir, f"valid_data_string.{alpha}.txt"), "w") as f:
            f.write(valid_string)

        test_strings = ["all_test:"]
        for file, weight in file_weights:
            test_strings[0] += f" {weight} {test_split_string} {file},"
            language_code = file[prefix:suffix]
            test_strings.append(f"test_{language_code}: 1 {test_split_string} {file}")
        test_string = " ".join([finalize_dataset_string(test_string) for test_string in test_strings])
        with open(os.path.join(args.input_files_dir, f"test_data_string.{alpha}.txt"), "w") as f:
            f.write(test_string)
