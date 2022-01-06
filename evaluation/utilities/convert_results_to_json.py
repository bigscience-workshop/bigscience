import json
import math
import os
from argparse import ArgumentParser
from os import listdir
from os.path import isfile

def get_args():
    parser = ArgumentParser()
    # --experiments tr3d-1B3-oscar-checkpoints,tr3e-1B3-c4-checkpoints,tr3m-1B3-pile-checkpoints
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment we want to download.')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='Result directory containing all results, and to store aggregated json results.')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Experiment training batch size.')
    parser.add_argument('--sequence_length', type=int, default=2048,
                        help='Experiment training sequence length.')
    parser.add_argument('--rampup-batch-size', type=lambda s: tuple(int(item) for item in s.split(',')), default=(32, 32, 2_000_000),
                        help='Experiment training batch size rampup.')
    return parser.parse_args()

def checkpoint_step_to_tokens(checkpoint_step, args) -> int:
    def fn(checkpoint_step) -> int:
        if not hasattr(checkpoint_step_to_tokens, "CACHE"):
            checkpoint_step_to_tokens.CACHE = {}

        BATCH_SIZE=args.batch_size
        SEQUENCE_LENGTH=args.sequence_length
        # Linear increase in terms of samples.
        RAMPUP_BATCH_SIZE = args.rampup_batch_size

        # Compute RAMPUP checkpoint_step
        if not hasattr(checkpoint_step_to_tokens, "RAMPUP_OFFSET"):
            initial_batch_size, increment_batch_size, sample_limit_for_rampup = RAMPUP_BATCH_SIZE
            number_of_increments = (BATCH_SIZE - initial_batch_size) // increment_batch_size
            assert (BATCH_SIZE - initial_batch_size) % increment_batch_size == 0

            offset_step = 0
            start_sample = 0
            for incr in range(number_of_increments):
                batch_size = initial_batch_size + incr * increment_batch_size
                end_sample = int(math.ceil((incr + 1) * sample_limit_for_rampup / number_of_increments))
                number_of_step_per_increment = int(math.ceil((end_sample - start_sample) / batch_size))
                checkpoint_step_to_tokens.CACHE.update({
                    offset_step + i: (start_sample + i * batch_size) * SEQUENCE_LENGTH
                    for i in range(number_of_step_per_increment)
                })
                offset_step += number_of_step_per_increment
                start_sample += number_of_step_per_increment * batch_size

            checkpoint_step_to_tokens.CACHE[offset_step] = start_sample * SEQUENCE_LENGTH
            checkpoint_step_to_tokens.RAMPUP_OFFSET = offset_step

        if checkpoint_step in checkpoint_step_to_tokens.CACHE:
            return checkpoint_step_to_tokens.CACHE[checkpoint_step]

        number_steps_after_rampup = checkpoint_step - checkpoint_step_to_tokens.RAMPUP_OFFSET
        assert number_steps_after_rampup >= 0

        slope = BATCH_SIZE * SEQUENCE_LENGTH

        checkpoint_step_to_tokens.CACHE[checkpoint_step] = \
            checkpoint_step_to_tokens.CACHE[checkpoint_step_to_tokens.RAMPUP_OFFSET] + \
            slope * number_steps_after_rampup
        return checkpoint_step_to_tokens.CACHE[checkpoint_step]
    return fn(checkpoint_step)

def main():
    args = get_args()
    result_dir = args.result_dir
    experiment = args.experiment

    results_file_per_checkpoint = [
        file
        for file in listdir(result_dir)
        if isfile(os.path.join(result_dir, file)) and file.startswith(experiment)
    ]
    checkpoint_steps = sorted([int(file.split("_")[-1].split(".json")[0]) for file in results_file_per_checkpoint])
    absolute_paths = [f"{result_dir}/{experiment}_{checkpoint_step}.json" for checkpoint_step in checkpoint_steps]
    # format = "{EXPERIMENT_NAME}_{CHECKPOINT_STEP}.json"
    tokens = [checkpoint_step_to_tokens(checkpoint_step, args) for checkpoint_step in checkpoint_steps]

    result_json = {}
    for absolute_path in absolute_paths:
        with open(absolute_path, 'r') as fi:
            results = json.load(fi)["results"]

        for task in results:
            if task not in result_json:
                result_json[task] = {}

            for metric in results[task]:
                if metric not in result_json[task]:
                    result_json[task][metric] = []

                result_json[task][metric].append(results[task][metric])

    # check
    for task in result_json:
        assert len(tokens) == len(checkpoint_steps)
        for metric in result_json[task]:
            assert len(result_json[task][metric]) == len(checkpoint_steps)

    output_path = os.path.join(result_dir, f"{experiment}_agg.json")
    print(f"Printing results to {output_path}")
    with open(output_path, 'w') as fo:
        json.dump({"tokens": tokens, "checkpoints": checkpoint_steps, "results": result_json}, fo, indent=2)

if __name__ == "__main__":
    main()
