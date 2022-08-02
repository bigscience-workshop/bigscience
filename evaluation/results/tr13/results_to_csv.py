#!/usr/bin/env python

# this script converts results.json:
#
#   "results": {
#     "arc_challenge": {
#       "acc": 0.24232081911262798,
#       "acc_stderr": 0.01252159329580012,
#       "acc_norm": 0.2764505119453925,
#       "acc_norm_stderr": 0.013069662474252425
#     },
#
# into a format expected by a spreadsheet, which is:
#
#   task          metric   value    err
#   arc_challenge acc      xxx      yyy
#   arc_challenge acc_norm xxx      yyy
#   arc_challenge f1       xxx      yyy
#
# usage:
# report-to-csv.py results.json


import sys
import statistics
import json
import io
import csv

results_file = sys.argv[1]

csv_file = results_file.replace("json", "csv")

print(f"Converting {results_file} to {csv_file}")

with io.open(results_file, 'r', encoding='utf-8') as f:
    raw_results = json.load(f)

results = {}
for ds_name, v in sorted(raw_results.items()):
    results[ds_name.split("/")[-1]] = v

with io.open(csv_file, 'w', encoding='utf-8') as f:

    writer = csv.writer(f)
    writer.writerow(["dataset", "prompt", "metric", "value"])

    for ds_name, v in sorted(results.items()):
        acc_scores = []
        for prompt_name, res in sorted(v.items()):
            for metric, value in sorted(res["evaluation"].items()):
                writer.writerow([ds_name, prompt_name, metric, value])
                if metric == "accuracy":
                    acc_scores.append(value)
        if acc_scores:
            writer.writerow([ds_name, "median", "accuracy", statistics.median(acc_scores)])
