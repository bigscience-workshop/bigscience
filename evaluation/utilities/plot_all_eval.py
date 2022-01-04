import json
import os
from argparse import ArgumentParser

from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input-files', type=lambda s: s.split(','), required=True, help='Input files that hold all evaluation metrics')
    return parser.parse_args()

def main():
    args = get_args()

    plots = {} # {"{EVALUATION}_{METRIC}": plt.figure}
    for input_file in args.input_files:
        assert os.path.basename(input_file).endswith("_agg.json")
        experiment_name = os.path.basename(input_file).split("_agg.json")[0]
        with open(input_file, "r") as fi:
            experiment = json.load(fi)

        tokens = experiment["tokens"]
        for evaluation_name, evaluation in experiment["results"].items():
            for metric_name, metric in evaluation.items():
                key = f"{evaluation_name}_{metric_name}"
                if key[-7:] == "_stderr":
                    continue

                if key not in plots:
                    plot = plt.figure(len(plots))
                    plot = plot.add_subplot(1,1,1)
                    plot.set_title(key)
                    plots[key] = plot

                plot = plots[key]

                plot.plot(tokens, metric, label=experiment_name)

    for plot in plots.values():
        plot.legend()
    plt.show()

if __name__ == "__main__":
    main()
