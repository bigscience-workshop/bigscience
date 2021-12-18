import json
import math
from os import listdir
from os.path import isfile

def checkpoint_step_to_tokens(checkpoint_step) -> int:
    def fn(checkpoint_step) -> int:
        if not hasattr(checkpoint_step_to_tokens, "CACHE"):
            checkpoint_step_to_tokens.CACHE = {}

        BATCH_SIZE=512
        SEQUENCE_LENGTH=2048
        # Linear increase in terms of samples.
        RAMPUP_BATCH_SIZE = (32, 32, 2_000_000)

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
    path_base = "/gpfsscratch/rech/six/commun/synched_exps/eval-tr3/results"
    output_path = f"{path_base}/final.json"
    experiments = {
        "tr3d-1B3-oscar-checkpoints",
        "tr3e-1B3-c4-checkpoints",
        "tr3m-1B3-pile-checkpoints"
    }

    # We merge all results in a single json
    merged_json = {}
    for experiment in experiments:
        results_file_per_checkpoint = [
            file
            for file in listdir(path_base)
            if isfile(f"{path_base}/{file}") and file.startswith(experiment)
        ]
        checkpoint_steps = sorted([int(file.split("_")[-1].split(".json")[0]) for file in results_file_per_checkpoint])
        absolute_paths = [f"{path_base}/{experiment}_{checkpoint_step}" for checkpoint_step in checkpoint_steps]
        # format = "{EXPERIMENT_NAME}_{CHECKPOINT_STEP}.json"
        tokens = [checkpoint_step_to_tokens(checkpoint_step) for checkpoint_step in checkpoint_steps]

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

        merged_json[experiment] = {
            "tokens": tokens,
            "checkpoints": checkpoint_steps,
            "results": result_json
        }

    final_results = merged_json
    with open(output_path, 'w') as fo:
        json.dump(final_results, fo)

if __name__ == "__main__":
    main()
