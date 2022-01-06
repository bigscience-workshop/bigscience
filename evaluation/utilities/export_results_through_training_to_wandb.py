import os

import numpy as np
import wandb
import json
import argparse

RANDOM_BASELINE={
    "arc_challenge": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "arc_easy": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "boolq": 0.5,
    "copa": 0.5,
    "headqa_en": 0.25,
    "hellaswag": 0.25,
    "lambada": 0., # Safe to say that random models won't perform well at all.
    "logiqa": 0.25,
    "mathqa": (4360 * 1/ 5 - (4475 - 4360) * 1/ 4) / 4475,
    "mrpc": 0.5,
    "multirc": 0., # TODO: I couldn't figure it out
    "openbookqa": 0.25,
    "piqa": 0.5,
    "prost": 0.25,
    "pubmedqa": 1/3,
    "qnli": 0.5,
    "qqp": 0.5,
    "race": 0.25, # Source: https://arxiv.org/pdf/1704.04683.pdf table 5
    "rte": 0.5,
    "sciq": 0.25,
    "sst": 0.5,
    "triviaqa": 0.,
    "webqs": 0.,
    "wic": 0.5,
    "winogrande": 0.5,
    "wnli": 0.5,
    "wsc": 0.5
}

def normalise(score, task):
    return (score - RANDOM_BASELINE[task]) / (1. - RANDOM_BASELINE[task])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, required=True)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--naive_average", action="store_true")
    parser.add_argument("--acc_average", action="store_true")
    parser.add_argument("--normalised_acc_average", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    for input_file in args.input_files:
        assert os.path.basename(input_file).endswith("_agg.json")
        experiment_name = os.path.basename(input_file).split("_agg.json")[0]
        with open(input_file, "r") as fi:
            experiment = json.load(fi)

        results = experiment["results"]
        tokens = experiment["tokens"]
        run = wandb.init(project="bigscience-tr3-evaluation-through-training", entity="timerobber", name=experiment_name,
                         reinit=True)
        for i, n_tokens in enumerate(tokens):
            all_values = []
            acc_average = []
            normalised_acc_average = []
            for task, task_results in results.items():
                values = None
                for metric, values in task_results.items():
                    if args.all_tasks:
                        wandb.log({f"{task}_{metric}": values[i], "tokens": tokens[i]})
                    if "stderr" not in metric and "ppl" not in metric:
                        all_values.append(values[i])
                        if metric == "acc":
                            acc_average.append(values[i])
                            normalised_acc_average.append(normalise(values[i], task))
            if args.naive_average:
                wandb.log({f"naive_average": np.mean(all_values), "tokens": tokens[i]})
            if args.acc_average:
                wandb.log({f"acc_average": np.mean(acc_average), "tokens": tokens[i]})
            if args.normalised_acc_average:
                wandb.log({f"normalised_acc_average": np.mean(normalised_acc_average), "tokens": tokens[i]})

        run.finish()

if __name__ == "__main__":
    main()
