import argparse
import os
from typing import List, Dict
from datasets import Dataset, load_dataset
from multiprocessing import cpu_count


def get_size_per_example(texts: List[str]) -> Dict:
    size_values = [len(text.encode()) for text in texts]
    examples = {"bytes_len": size_values}
    return examples


def full_size_estimation(
    ds: Dataset,
    batch_size: int,
    content_key: str = "text",
    num_proc: int = cpu_count(),
) -> int:
    if len(ds) == 0:
        return 0

    ds_with_size = ds.map(
        get_size_per_example,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        input_columns=[content_key],
        remove_columns=ds.column_names,
    )
    len_bytes = sum(ds_with_size["bytes_len"])
    return len_bytes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="path to jsonl file containing the data",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="path to jsonl file containing the data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds = load_dataset("json", data_files=args.input_path, split="train")
    size = full_size_estimation(ds, batch_size=32)
    dataset_name = os.path.basename(args.input_path)[:-6]
    with open(os.path.join(args.output_folder, dataset_name), "w") as f:
        f.write(str(size))
