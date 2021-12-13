import datasets
import json

steps_vs_samples = datasets.load_dataset("csv", data_files="run-.-tag-steps-vs-samples_y=steps,x=samples.csv")["train"]

slope = (steps_vs_samples[-1]["Step"] - steps_vs_samples[-2]["Step"]) / (
        steps_vs_samples[-1]["Value"] - steps_vs_samples[-2]["Value"])
offset = steps_vs_samples[-1]["Step"] - steps_vs_samples[-1]["Value"] * slope

token_interval = 1e10
step_interval = 1500
tokens_per_sample = 2048
token_count = token_interval

output_checkpoints = []

for item in steps_vs_samples:
    if item["Step"] * tokens_per_sample > token_count:
        token_count += token_interval
        step = step_interval * (item['Value'] // step_interval)
        tokens = tokens_per_sample * (slope * (step_interval * (item['Value'] // step_interval)) + offset)
        print(f"step: {step}")
        print(f"tokens at that step: {tokens}")
        output_checkpoints.append({"step": step, "tokens": tokens})


json.dump(output_checkpoints, open("steps_to_evaluate_with_tokens.json", "w"))
