import numpy as np
import wandb
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--naive_average", action="store_true")
    parser.add_argument("--acc_average", action="store_true")
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
            acc_average = []
            for task, task_results in results.items():
                values = None
                for metric, values in task_results.items():
                    if args.all_tasks:
                        wandb.log({f"{task}_{metric}": values[i], "tokens": tokens[i]})
                    if "stderr" not in metric and "ppl" not in metric:
                        all_values.append(values[i])
                        if metric == "acc":
                            acc_average.append(values[i])
            if args.naive_average:
                wandb.log({f"naive_average": np.mean(all_values), "tokens": tokens[i]})
            if args.acc_average:
                wandb.log({f"acc_average": np.mean(acc_average), "tokens": tokens[i]})

        run.finish()
