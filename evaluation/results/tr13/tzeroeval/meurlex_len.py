import json
import os
import sys

dir = sys.argv[1]
paths = os.listdir(dir)
len_dict = {}

for path in paths:
    if not(path.startswith("examples")):
        continue
    pred_lens = []
    tar_lens = []
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            pred_lens.append(len(ex["pred"]))
            tar_lens.append(len(ex["target"][0]))
    
    avg_pred_len = sum(pred_lens) / len(pred_lens)
    tar_pred_len = sum(tar_lens) / len(tar_lens)
    print(avg_pred_len, tar_pred_len)

    len_dict.setdefault(path, {})
    len_dict[path]["pred"] = avg_pred_len
    len_dict[path]["tar"] = tar_pred_len


print("Average Pred: ", sum(len_dict[k]["pred"] for k in len_dict) / len(len_dict))
print("Average Target: ", sum(len_dict[k]["tar"] for k in len_dict) / len(len_dict))

with open("meurlex_lens.json", "w") as f:
    json.dump(len_dict, f)
