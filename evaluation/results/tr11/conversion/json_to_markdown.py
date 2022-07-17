"""
Table example:

| Task | Language | Metric | BLOOM-176B | OPT-176B |
|:--------|:-----------------|:------------------------|-------------:|------------:|
| arc_challenge | eng | acc | 0.4112627986348123 | 0.4121160409556314 |


Metadata example:

model-index:
- name: bart-large-cnn-samsum
  results:
    - task:
      type: summarization
      name: Summarization
    dataset:
      name: 'SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization'
      type: samsum
    metrics:
    - name: Validation ROGUE-1
      type: rogue-1
      value: 42.621
    - name: Validation ROGUE-2
      type: rogue-2
      value: 21.9825
    - name: Validation ROGUE-L
      type: rogue-l
      value: 33.034
    - name: Test ROGUE-1
      type: rogue-1
      value: 41.3174
    - name: Test ROGUE-2
      type: rogue-2
      value: 20.8716
    - name: Test ROGUE-L
      type: rogue-l
      value: 32.1337
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
    metrics:
    - name: ROUGE-1
      type: rouge
      value: 41.3282
      verified: true
    - name: ROUGE-2
      type: rouge
      value: 20.8755
      verified: true
    - name: ROUGE-L
      type: rouge
      value: 32.1353
      verified: true
    - name: ROUGE-LSUM
      type: rouge
      value: 38.401
      verified: true
    - name: loss
      type: loss
      value: 1.4297215938568115
      verified: true
    - name: gen_len
      type: gen_len
      value: 60.0757
      verified: true
"""

import json
import statistics

FILE_NAMES = ["bslmeval", "humaneval_temp02", "humaneval_temp06", "humaneval_temp08"]

# Optionally subselect tasks
SELECTED_LIST = [
    "winogrande"
]

with open("bloom2b5/bslmeval.json", "r") as f:
    bloom_bslmeval = json.load(f)

with open("opt/bslmeval.json", "r") as f:
    opt_bslmeval = json.load(f)



results_formatted = {}
for task_name in bloom_bslmeval["results"]:
    #if task_name not in SELECTED_LIST:
    #    continue
    date_keys = list(bloom_bslmeval["results"][task_name].keys())
    assert len(date_keys) == 1
    metrics = bloom_bslmeval["results"][task_name][date_keys[0]]

    lang = "eng"
    if "gsarti/flores_101_" in task_name:
        lang = task_name.replace("gsarti/flores_101_", "").replace("+null", "")
    elif "lambada_mt_de" in task_name:
        lang = "deu"
    elif "lambada_mt_en" in task_name:
        lang = "eng"
    elif "lambada_mt_es" in task_name:
        lang = "esp"
    elif "lambada_mt_it" in task_name:
        lang = "ita"
    elif "lambada" == task_name:
        continue
    elif "crows_pairs_french" in task_name:
        lang = "fra"
    elif "headqa" == task_name:
        lang = "esp"

    if "acc" in metrics:
        main_metric_name = "acc ↑"
    elif "byte_perplexity" in metrics:
        main_metric_name = "byte_perplexity ↓"
    elif "pass@100" in metrics:
        main_metric_name = "pass@100 ↑"
    elif "em" in metrics:
        main_metric_name = "em ↑"

    date_keys_opt = list(opt_bslmeval["results"][task_name].keys())
    score_opt = opt_bslmeval["results"][task_name][date_keys_opt[0]][main_metric_name[:-2]]

    fin_task_name = metrics.get("task_name", task_name)
    
    results_formatted.setdefault(fin_task_name, {})
    results_formatted[fin_task_name].setdefault("prompts", [])
    results_formatted[fin_task_name].setdefault("all_metrics", [])
    results_formatted[fin_task_name].setdefault("main_metrics", [])

    if "prompt_name" in metrics:
        results_formatted[fin_task_name]["prompts"].append(metrics["prompt_name"])
    results_formatted[fin_task_name]["name"] = fin_task_name
    results_formatted[fin_task_name]["lang"] = lang
    results_formatted[fin_task_name]["all_metrics"].append(metrics) # [{name: score}]
    results_formatted[fin_task_name]["main_metrics"].append((main_metric_name, metrics[main_metric_name[:-2]], score_opt))
    results_formatted[fin_task_name]["type"] = "text-generation"

# Take Median of scores
for k, v in results_formatted.items():
    if "prompts" in v and len(v["prompts"]) > 1:
        assert len(v["all_metrics"]) == len(v["main_metrics"])
        num_scores = len(v["main_metrics"])

        bloom_median = statistics.median([triplet[1] for triplet in v["main_metrics"]])
        opt_median = statistics.median([triplet[2] for triplet in v["main_metrics"]])

        results_formatted[k]["main_metrics"] = [(
            v["main_metrics"][0][0],
            bloom_median,
            opt_median,
        )]

        results_formatted[k]["name"] = results_formatted[k]["name"] + f" (Median of {num_scores} prompts)"



def keep_best_score(new_eval, old_eval):
    for k, v in new_eval.items():
        old_eval[k] = max(old_eval[k], v) 
    return old_eval

for i, temp in enumerate(["02", "06", "08"]):
    with open(f"bloom/humaneval_temp{temp}.json", "r") as f:
        if i > 0:
            keep_best_score(json.load(f), bloom_humaneval)
        else:
            bloom_humaneval = json.load(f)
    with open(f"opt/humaneval_temp{temp}.json", "r") as f:
        if i > 0:
            keep_best_score(json.load(f), opt_humaneval)
        else:
            opt_humaneval = json.load(f)

results_formatted["humaneval"] = {
    "name": "humaneval",
    "lang": "python",
    "all_metrics": [bloom_humaneval], # [{name: score}]
    "main_metrics": [(f"{name} ↑", score, opt_humaneval[name]) for name, score in bloom_humaneval.items()],
    "type": "text-generation"
}



# Add multilingual average
for k, v in results_formatted.items():
    if "prompts" in v and len(v["prompts"]) > 1 and len(v["main_metrics"]) > 1:
        assert len(v["all_metrics"]) == len(v["main_metrics"]), f"{k}, {len(v['all_metrics'])}, {len(v['main_metrics'])}"
        num_scores = len(v["main_metrics"])

        bloom_median = statistics.median([triplet[1] for triplet in v["main_metrics"]])
        opt_median = statistics.median([triplet[2] for triplet in v["main_metrics"]])

        results_formatted[k]["main_metrics"] = [(
            v["main_metrics"][0][0],
            bloom_median,
            opt_median,
        )]

        results_formatted[k]["name"] = results_formatted[k]["name"] + f" (Median of {num_scores} prompts)"

"""Optional aggregated statistics
bloom_mean = statistics.mean([triplet[1] for k,v in results_formatted.items() for triplet in v["main_metrics"] if v["lang"] == "eng"])
opt_mean = statistics.mean([triplet[2] for k,v in results_formatted.items() for triplet in v["main_metrics"] if v["lang"] == "eng"])

results_formatted["mean_eng"] = {
    "name": "mean_eng ↑",
    "lang": "eng",
    "all_metrics": [{"mean": bloom_mean}], # [{name: score}]
    "main_metrics": [("mean", bloom_mean, opt_mean)],
    "type": "text-generation"
}

bloom_mean = statistics.mean([triplet[1] for k,v in results_formatted.items() for triplet in v["main_metrics"] if "flores" in k])
opt_mean = statistics.mean([triplet[2] for k,v in results_formatted.items() for triplet in v["main_metrics"] if "flores" in k])

results_formatted["mean_multilingual"] = {
    "name": "mean_multilingual (Flores) ↓",
    "lang": "mul",
    "all_metrics": [{"mean": bloom_mean}], # [{name: score}]
    "main_metrics": [("mean", bloom_mean, opt_mean)],
    "type": "text-generation"
}

main_metrics = ([triplet for k,v in results_formatted.items() for triplet in v["main_metrics"]])

bloom_best_on, opt_best_on = 0,0
for (name, bloom, opt) in main_metrics:
    if name[:-2] in ["acc", "em"] or "pass" in name:
        if bloom > opt:
            bloom_best_on += 1
        elif bloom < opt:
            opt_best_on += 1
    elif name[:-2] in ["byte_perplexity"]:
        if bloom < opt:
            bloom_best_on += 1
        elif bloom > opt:
            opt_best_on += 1
"""
### Markdown Table ###

HEADER = "| Task | Language | Metric | BLOOM-350M | BLOOM-750M | BLOOM-1B3 | BLOOM-2B5 | BLOOM-6B3 | BLOOM-176B |"
SEP = "|:----|:----|:----|:----:|"
ONE_LINE = "| {} | {} | {} | {} |"

TABLE_STRING = "\n".join([HEADER, SEP])

for task_name, res_dict in results_formatted.items():
    for (name, score, score_opt) in res_dict["main_metrics"]:
        TABLE_STRING += "\n" + ONE_LINE.format(
            res_dict["name"],
            res_dict["lang"],
            name,
            round(score, 3),
            round(score_opt, 3),
        )

with open("./mdtable.txt", "w") as f:
    f.write(TABLE_STRING)



### Metadata ###

HEADER = "model-index:"
MODEL = "- name: bloom"
RES = "  results:"

META_STRING = "\n".join([HEADER, MODEL, RES])

ONE_TASK = "  - task:\n      type: {}\n      name: {}\n    dataset:\n      name: {}\n      type: {}\n    metrics:"
ONE_METRIC = "    - name: {}\n      type: {}\n      value: {}\n      verified: false"

for task_name, res_dict in results_formatted.items():
    META_STRING += "\n" + ONE_TASK.format(
        res_dict["type"],
        res_dict["type"].replace("-", " "),
        task_name,
        task_name,
    )
    for (name, score, score_opt) in res_dict["main_metrics"]:
            META_STRING += "\n" + ONE_METRIC.format(
                name.split(" ")[0],
                name.split(" ")[0],
                score
            )   
"""
    for metrics in res_dict["all_metrics"]:
        for metric_name, metric in metrics.items():
            if isinstance(metric, str):
                continue
            META_STRING += "\n" + ONE_METRIC.format(
                metric_name,
                metric_name,
                metric
            )   
"""


with open("./mdmeta.txt", "w") as f:
    f.write(META_STRING)
