import json

def main():
    path_base = "/gpfsscratch/rech/six/commun/synched_exps/eval-tr3/results"
    output_path = f"{path_base}/final.json"
    experiments = [
        "tr3d-1B3-oscar-checkpoints",
        "tr3e-1B3-c4-checkpoints",
        "tr3m-1B3-pile-checkpoints"
    ]
    # It's very important we keep the ordering.
    checkpoint_steps = [
        19500,
        28500,
        37500,
        48000,
        57000,
        66000,
        76500,
        85500,
        94500,
        105000,
        114000
    ]
    # Tokens were added by hand to the final json.
    tokens = [10044178432, 19481362432, 28918546432, 39928594432, 49365778432, 58802962432, 69813010432, 79250194432, 88687378432, 99697426432, 109134610432]
    assert len(tokens) == len(checkpoint_steps)

    # We merge all results in a single json
    merged_json = {}
    for experiment in experiments:
        merged_json[experiment] = {}
        for ckpt_step in checkpoint_steps:
            absolute_path = f"{path_base}/{experiment}_{ckpt_step}.json"
            with open(absolute_path, 'r') as fi:
                results = json.load(fi)["results"]

            for task in results:
                if task not in merged_json[experiment]:
                    merged_json[experiment][task] = {}

                for metric in results[task]:
                    if metric not in merged_json[experiment][task]:
                        merged_json[experiment][task][metric] = []

                    merged_json[experiment][task][metric].append(results[task][metric])

    # check
    for experiment in merged_json:
        for task in merged_json[experiment]:
            for metric in merged_json[experiment][task]:
                assert len(merged_json[experiment][task][metric]) == len(checkpoint_steps)

    final_results = {"tokens": tokens, "checkpoint_steps": checkpoint_steps, "results": merged_json}
    with open(output_path, 'w') as fo:
        json.dump(final_results, fo)

if __name__ == "__main__":
    main()
