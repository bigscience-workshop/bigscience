import csv
import json
import re
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

"""
Plot results per (dataset_name, dataset_config_name).
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--json_paths", nargs="+", type=str, help="Json files to plot together", required=True)
    parser.add_argument("--t0_csv_path", type=str, help="T0 eval results path")
    args = parser.parse_args()

    return args

def load_t0_results(csv_path):
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))

def load_json(json_path):
    with open(json_path, "r") as fi:
        return json.load(fi)

def get_experiment_name(filename: str):
    name = re.sub(r"_([0-9]*)$", r" [\1]", filename)
    name = name.replace("span_corruption", "SC")
    name = re.sub(r"^enc_dec", "ED", name)
    name = re.sub(r"^nc_dec", "NCD", name)
    name = re.sub(r"^c_dec", 'CD', name)
    name = name.replace("full_lm", "FLM")
    name = name.replace("prefix_lm", "PLM")
    name = re.sub(r"t0_adapt_([0-9]+)", r"T0(\1)", name)
    if name[:3] == "CD_":
        name = re.sub(r"lm_adapt_([0-9]+)", r"FLM(\1)", name)
        name = re.sub(r"t0_adapt_nc_([0-9]+)", r"T0 AS NC (\1)", name)
        name = re.sub(r"nc_sc_([0-9]+)", r"SC as NC(\1)", name)
        name = re.sub(r"nc_t0_([0-9]+)", r"T0 as NC(\1)", name)
    elif name[:4] == "NCD_" or name[:3] == "ED_":
        if "flm_adapt" in name:
            name = re.sub(r"flm_adapt_([0-9]+)", r"FLM AS CD(\1)", name)
        else:
            name = re.sub(r"lm_adapt_([0-9]+)", r"PLM(\1)", name)
    else:
        raise NotImplementedError
    name = name.replace("_", " + ")
    return name

TASKS = {
    # T0  evaluation
    "super_glue_copa": ("COPA", 0.5),
    "anli_dev_r1": ("ANLI R1", 1/3),
    "anli_dev_r2": ("ANLI R2", 1/3),
    "anli_dev_r3": ("ANLI R3", 1/3),
    "super_glue_cb": ("CB", 1/3),
    "super_glue_rte": ("RTE", 0.5),
    "super_glue_wsc.fixed": ("WSC", 0.5),
    "winogrande_winogrande_xl": ("Winogrande", 0.5),
    "super_glue_wic": ("WiC", 0.5),
    "hellaswag": ("HellaSwag", 0.25),
    "story_cloze_2016": ("StoryCloze", 0.5),

    # XNLI evaluation
    "xnli_ar": ("XNLI ar (en prompts)", 1/3),
    "xnli_bg": ("XNLI bg (en prompts)", 1/3),
    "xnli_de": ("XNLI de (en prompts)", 1/3),
    "xnli_el": ("XNLI el (en prompts)", 1/3),
    "xnli_en": ("XNLI en (en prompts)", 1/3),
    "xnli_es": ("XNLI es (en prompts)", 1/3),
    "xnli_fr": ("XNLI fr (en prompts)", 1/3),
    "xnli_hi": ("XNLI hi (en prompts)", 1/3),
    "xnli_ru": ("XNLI ru (en prompts)", 1/3),
    "xnli_sw": ("XNLI sw (en prompts)", 1/3),
    "xnli_th": ("XNLI th (en prompts)", 1/3),
    "xnli_tr": ("XNLI tr (en prompts)", 1/3),
    "xnli_ur": ("XNLI ur (en prompts)", 1/3),
    "xnli_vi": ("XNLI vi (en prompts)", 1/3),
    "xnli_zh": ("XNLI zh (en prompts)", 1/3),
}

def plot(mtf_data, t0_data):
    args = get_args()

    assert len(TASKS) == 26
    fig, axs = plt.subplots(3, 9, figsize=(20, 5))
    axs = axs.flatten()

    task_min_score = {}
    task_max_score = {}
    task_median_score = {}
    for n, (task, (task_name, random_baseline)) in enumerate(TASKS.items()):
        # Normalising names
        mtf_task = task
        t0_task = task
        if task.startswith("anli_dev_r"):
            t0_task = re.sub("dev_", "", task)
        elif task == "hellaswag":
            mtf_task = "hellaswag_None"

        t5lm_scores = [float(r["score"]) for r in t0_data
                       if r["runs"] == "xxl-lm-d4-091621"
                       and r["dataset_name"] == t0_task
                       and r["metric_name"] == "accuracy (Rank)"
                       and r["score"]]
        t0_scores = [float(r["score"]) for r in t0_data
                     if r["runs"] == "xxl-lm-d4-091621-512"
                     and r["dataset_name"] == t0_task
                     and r["metric_name"] == "accuracy (Rank)"
                     and r["score"]]

        mtf_scores = [
            (
                name,
                [100 * value["evaluation"]["accuracy"] for prompt, value in data[mtf_task].items()]
                if mtf_task in data else
                []
            )
            for name, data in mtf_data.items()
        ]

        all_experiment_scores_with_name = [("T5 + LM", t5lm_scores), ("T0", t0_scores), *mtf_scores]
        # Plot
        axs[n].axhline(100 * random_baseline, 0, len(all_experiment_scores_with_name), label="Random")
        for i, (exp_name, scores) in enumerate(all_experiment_scores_with_name):
            axs[n].scatter([i] * len(scores), scores, s=50, alpha=0.4, label=exp_name)
        axs[n].set_title(task_name, fontsize=8)

        # # Gather median values
        # task_min_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.min(scores)) for (exp_name, scores) in all_experiment_scores_with_name]
        # task_max_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.max(scores)) for (exp_name, scores) in all_experiment_scores_with_name]
        # task_median_score[task] = [("Random", 100 * random_baseline)] + [(exp_name, np.median(scores)) for (exp_name, scores) in all_experiment_scores_with_name]

    last_ax_id = len(TASKS) - 1
    axs[last_ax_id].legend(bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs[last_ax_id + 1:]:
        ax.set_visible(False)

    # if args.aggregated_results:
    #     # ====== Plot agregated values =======
    #     fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    #     axs = axs.flatten()
    #     last_ax_id=0
    #     experiment_names = [elt[0] for elt in next(iter(task_median_score.values()))]
    #
    #     def plot_scores_with_name(median_score_with_name, max_score, min_score, ax, title):
    #         assert len(median_score_with_name) == len(max_score) and len(median_score_with_name) == len(min_score)
    #         ax.axhline(
    #             median_score_with_name[0][1],
    #             0, len(median_score_with_name) - 1,
    #             label=median_score_with_name[0][0]
    #         )
    #         for i, ((name, median_score), max_score, min_score) in enumerate(zip(median_score_with_name[1:], max_score[1:], min_score[1:])):
    #             ax.errorbar(
    #                 i, median_score, ((median_score - min_score,), (max_score - median_score,)),
    #                 fmt="o", elinewidth=1, label=name)
    #         ax.set_title(title)
    #
    #     def get_average_normalised_score(task_scores):
    #         normalised_scores = []
    #         for scores_with_name in task_scores.values():
    #             random_name, random_baseline = scores_with_name[0]
    #             assert random_name == "Random"
    #             normalised_scores_per_task = [(scores - random_baseline) / (100 - random_baseline) for _, scores in
    #                                           scores_with_name]
    #             normalised_scores.append(normalised_scores_per_task)
    #         return np.mean(normalised_scores, axis=0)
    #
    #     def get_average_score(task_scores):
    #         return np.mean(
    #             [[scores for _, scores in scores_with_name] for scores_with_name in task_scores.values()], axis=0)
    #
    #     # Plot average task score
    #     average_task_median_score = get_average_score(task_median_score)
    #     assert len(experiment_names) == len(average_task_median_score)
    #     average_task_media_score_with_name = list(zip(experiment_names, average_task_median_score))
    #     del average_task_median_score
    #     plot_scores_with_name(
    #         median_score_with_name=average_task_media_score_with_name,
    #         max_score=get_average_score(task_max_score),
    #         min_score=get_average_score(task_min_score),
    #         ax=axs[last_ax_id],
    #         title=f"Average of task median scores"
    #     )
    #     last_ax_id += 1
    #
    #     # Plot average of task median normalised scores `normalised_score = (score - random) / (1 - random)`
    #     average_task_normalised_median_score = get_average_normalised_score(task_median_score)
    #     assert len(experiment_names) == len(average_task_normalised_median_score)
    #     average_task_normalised_median_score_with_name = list(
    #         zip(experiment_names, average_task_normalised_median_score))
    #     del average_task_normalised_median_score
    #     plot_scores_with_name(
    #         median_score_with_name=average_task_normalised_median_score_with_name,
    #         max_score=get_average_normalised_score(task_max_score),
    #         min_score=get_average_normalised_score(task_min_score),
    #         ax=axs[last_ax_id],
    #         title=f"Average of task normalised median scores"
    #     )
    #     last_ax_id += 1
    #
    #     axs[last_ax_id -1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    #     for ax in axs[last_ax_id:]:
    #         ax.set_visible(False)


def main():
    args = get_args()

    # Load results
    t0_data = load_t0_results(args.t0_csv_path)
    mtf_data = {
        re.sub(".json", "", json_path): load_json(json_path)
        for json_path in args.json_paths
    }

    plot(mtf_data, t0_data)

    plt.show()
    print("Finished")

if __name__ == "__main__":
    main()
