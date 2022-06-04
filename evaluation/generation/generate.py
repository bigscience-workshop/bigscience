import argparse
import json

from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer, AutoConfig, logging

logging.set_verbosity_debug()

def get_args():
    parser = argparse.ArgumentParser()
    # --checkpoint /gpfswork/rech/six/uan68tv/model-conversion/main-gs-47400-transformers-sharded
    # --checkpoint /gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-555750
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--parallelize-for-176b", action="store_true")
    parser.add_argument("--generate-max-length", type=int, default=50, help="max generation length")
    parser.add_argument("--greedy", action="store_true")

    return parser.parse_args()

def generate_from_text(model, text, tokenizer, max_length=200, greedy=False):
    input_ids = tokenizer.encode(text, return_tensors='pt').to("cuda:0")
    max_length = input_ids.size(-1) + max_length
    if greedy:
        greedy_output = model.generate(input_ids.to('cuda:0'), max_length=max_length)
    else:
        greedy_output = model.generate(input_ids.to('cuda:0'), max_length=max_length, do_sample=True, top_k=0)
    return {
        "inputs": text,
        "outputs": tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    }

def main():
    args = get_args()

    model = BigScience176BLMHeadModel.from_pretrained(args.checkpoint, use_cache=False, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    print("successfully loaded model")

    if args.parallelize_for_176b:
        device_map = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            1: [9, 10, 11, 12, 13, 14, 15, 16, 17],
            2: [18, 19, 20, 21, 22, 23, 24, 25, 26],
            3: [27, 28, 29, 30, 31, 32, 33, 34, 35],
            4: [36, 37, 38, 39, 40, 41, 42, 43, 44],
            5: [45, 46, 47, 48, 49, 50, 51, 52, 53],
            6: [54, 55, 56, 57, 58, 59, 60, 61, 62],
            7: [63, 64, 65, 66, 67, 68, 69],
        }
        model.parallelize(device_map)
    else:
        model = model.cuda()
    model.eval()
    print("successfully parallelized model")

    while True:
        text = ''
        while True:
            dummy = input('''Enter the paragraph :''')+'\n'
            if dummy=='\n':
                break
            text += dummy
        output = generate_from_text(model, text, tokenizer, max_length=args.generate_max_length, greedy=args.greedy)
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
