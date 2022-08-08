import glob
import json
import os
import multiprocessing

jsonl_files = glob.glob("/gpfswork/rech/six/commun/bigscience-training/jsonls/xp3cappedmixed/*/*.jsonl")
print(jsonl_files)

#for path in jsonl_files:
def update_jsonl(path):
    print(path)
    with open(path, "r") as jsonl_file, open(path.replace(".jsonl", "tmp.jsonl"), "w") as jsonl_file_out:
        for line in jsonl_file:
            data = json.loads(line)
            data["targets"] = data["targets"][0]
            jsonl_file_out.write(json.dumps(data) + "\n")
    os.rename(path.replace(".jsonl", "tmp.jsonl"), path)


with multiprocessing.Pool(processes=multiprocessing.cpu_count()-5) as pool:
    pool.map(update_jsonl, jsonl_files)
