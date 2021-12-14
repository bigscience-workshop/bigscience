import json
from argparse import ArgumentParser

from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True, help='Input file that hold all evaluation metrics')
    return parser.parse_args()

def main():
    args = get_args()

    with open(args.input_file, "r") as fi:
        final = json.load(fi)

    tokens = final["tokens"]
    ckpt_steps = final["checkpoint_steps"]
    plots = {} # {"{EVALUATION}_{METRIC}": plt.figure}
    for experiment_name in final["results"]:
        for evaluation_name in final["results"][experiment_name]:
            for metric_name in final["results"][experiment_name][evaluation_name]:
                key = f"{evaluation_name}_{metric_name}"
                if key[-7:] == "_stderr":
                    continue

                if key not in plots:
                    plot = plt.figure(len(plots))
                    plot = plot.add_subplot(1,1,1)
                    plot.set_title(key)
                    plots[key] = plot

                plot = plots[key]

                plot.plot(tokens, final["results"][experiment_name][evaluation_name][metric_name], label=experiment_name)

    for plot in plots.values():
        plot.legend()
    plt.show()

if __name__ == "__main__":
    main()
