import json
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input-files', type=lambda s: s.split(','), required=True, help='Input file that hold all evaluation metrics')
    return parser.parse_args()

# TODO: fill it up
RANDOM_BASELINE={
    "arc_challenge_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "arc_easy_acc": 0.2502, # Source: https://arxiv.org/pdf/1803.05457.pdf table 6
    "boolq_acc": 0.5,
    "copa_acc": 0.5,
    "headqa_acc": 0.25, # TODO: That's a pain as some have 4, some have 5 and nobody reports random baseline
    "hellaswag_acc": 0.25,
    "lambada_acc": 0., # Safe to say that random models won't perform well at all.
    "logiqa_acc": 0.25,
    "mathqa_acc": 0.25, # TODO: That's a pain as some have 4, some have 5 and nobody reports random baseline
    "mrpc_acc": 0.5,
    "multirc_acc": 0., # TODO: I couldn't figure it out
    "openbookqa_acc": 0.25,
    "piqa_acc": 0.5,
    "prost_acc": 0.25,
    "pubmedqa_acc": 1/3,
    "qnli_acc": 0.5,
    "qqp_acc": 0.5,
    "race_acc": 0.25, # Source: https://arxiv.org/pdf/1704.04683.pdf table 5
    "rte_acc": 0.5,
    "sciq_acc": 0.25,
    "sst_acc": 0.5,
    "triviaqa_acc": 0.,
    "webqs_acc": 0.,
    "wic_acc": 0.5,
    "winogrande_acc": 0.5,
    "wnli_acc": 0.5,
    "wsc_acc": 0.5
}
def normalise_scores(scores_per_task):
    normalised_scores = {}
    for key,value in scores_per_task.items():
        # We assume it exists, otherwise we need to figure out what the random baseline is
        normalised_scores[key] = (value - RANDOM_BASELINE[key]) / (1. - RANDOM_BASELINE[key])
    # TODO: we need to substract the random baseline.
    return scores_per_task

def main():
    args = get_args()

    final = {}
    for input_file in args.input_files:
        assert os.path.basename(input_file).endswith("_agg.json")
        experiment_name = os.path.basename(input_file).split("_agg.json")[0]
        with open(input_file, "r") as fi:
            final[experiment_name] = json.load(fi)

    # We search for matching tokens
    matching_tokens = set(next(iter(final.values()))["tokens"])
    for experiment_name, experiment in final.items():
        tokens = experiment["tokens"]
        matching_tokens = matching_tokens & set(tokens)
        # Make sure we don't override existing data
        assert "token2checkpoint_step" not in experiment
        experiment["token2checkpoint_step"] = {token: ckpt_step for token, ckpt_step in zip(tokens, experiment["checkpoints"])}
        # Make sure we don't override existing data
        assert "token2id" not in experiment
        experiment["token2id"] = {token: _id for _id, token in enumerate(tokens)}
    matching_tokens = sorted(matching_tokens)
    print(f"Plotting only for tokens in {matching_tokens}")

    plots_per_keys = {}

    for token in matching_tokens:
        for experiment_name, experiment in final.items():
            _id = experiment["token2id"][token]
            scores_per_task = {
                "Average_acc": {
                    f"{evaluation_name}_{metric_name}": metric[_id]
                    for evaluation_name, evaluation in experiment["results"].items()
                    for metric_name, metric in evaluation.items()
                    if metric_name == "acc"
                },
                # "Average": {
                #     metric_name: values[i]
                #     for evaluation_name in final["results"][experiment_name]
                #     for metric_name, values in final["results"][experiment_name][evaluation_name].items()
                #     if metric_name[-7:] != "_stderr"
                # }
            }

            # Build plot graphs
            for key in scores_per_task:
                if key not in plots_per_keys:
                    plots_per_keys[key] = {}

                plot_per_token = plots_per_keys[key]
                if token in plot_per_token:
                    continue

                plot = plt.figure()
                plot = plot.add_subplot(1, 1, 1)
                plot.set_title(f"{key} - Number of tokens seen: {token}")
                plot_per_token[token] = plot

            # Plot per steps
            for key in plots_per_keys:
                scores = scores_per_task[key]
                plot = plots_per_keys[key][token]

                # Normalize score
                normalised_scores = normalise_scores(scores)

                # Sort scores, we order them from smalles to biggest
                sorted_scores = sorted(normalised_scores.values())

                # Compute the number of task over that sorted_scores.
                y = np.arange(len(sorted_scores), 0, -1) / len(sorted_scores)

                plot.step(x=sorted_scores, y=y, label=experiment_name)

    for plots in plots_per_keys.values():
        assert len(plots) == len(matching_tokens)
        for plot in plots.values():
            plot.legend()
    plt.show()

if __name__ == "__main__":
    main()
