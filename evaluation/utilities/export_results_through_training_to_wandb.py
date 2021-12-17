import numpy as np
import wandb
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--naive_average", action="store_true")
    parser.add_argument("--one_per_task_average", action="store_true")
    args = parser.parse_args()
    with open(args.input_file) as f:
        data = json.load(f)
    for experiment_name, experiment in data.items():
        results = experiment["results"]
        tokens = experiment["tokens"]
        run = wandb.init(project="bigscience-evaluation-through-training", entity="flukeellington", name=experiment_name,
                         reinit=True)
        for i, n_tokens in enumerate(tokens):
            all_values = []
            one_value_per_task = []
            for task, task_results in results.items():
                values = None
                added_this_round = False
                for metric, values in task_results.items():
                    if args.all_tasks:
                        wandb.log({f"{task}_{metric}": values[i], "tokens": tokens[i]})
                    if "stderr" not in metric and "ppl" not in metric:
                        added_this_round = True
                        all_values.append(values[i])
                if added_this_round:
                    one_value_per_task.append(all_values[-1])
            if args.naive_average:
                wandb.log({f"naive_average": np.mean(all_values), "tokens": tokens[i]})
            if args.one_per_task_average:
                wandb.log({f"one_per_task_average": np.mean(one_value_per_task), "tokens": tokens[i]})

        run.finish()
