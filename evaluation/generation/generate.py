import argparse
import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path", required=True)
    parser.add_argument("--max-memory-per-gpu", type=str, help="Defines maximum memory allocated to gpu", required=True)
    parser.add_argument("--global-step", type=str, default=None)
    parser.add_argument("--generate-max-length", type=int, default=50, help="max generation length")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.)
    parser.add_argument("--offload_folder", type=str, help="offload folder for accelerate", default="./offload")

    return parser.parse_args()

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def generate_from_text(model, text, tokenizer, max_length=200, greedy=False, top_k=0, top_p=0.):
    input_ids = tokenizer.encode(text, return_tensors='pt').to("cuda:0")
    max_length = input_ids.size(-1) + max_length
    
    greedy_output = model.generate(
        input_ids.to('cuda:0'),
        max_length=max_length,
        do_sample=not greedy,
        top_k=None if greedy else top_k,
        top_p=None if greedy else top_p
    )
    return tokenizer.decode(greedy_output[0], skip_special_tokens=True)

def main():
    args = get_args()
    print("Loading model")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")

    print("Loaded tokenizer!")
    start = datetime.datetime.now()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        max_memory=get_gpus_max_memory(args.max_memory_per_gpu),
        torch_dtype=torch.bfloat16,
        revision="gs{}".format(args.global_step) if args.global_step else None,
        offload_folder=args.offload_folder,
    )
    print(f"Loaded model in {datetime.datetime.now() - start}")

    texts = []
    while True:
        try:
            dummy = input('''Enter the paragraph (Enter for to validate new input line and Ctrl-c to start generating the prompt):''')
            texts.append(dummy)
        except KeyboardInterrupt:
            text = "\n".join(texts)
            output = generate_from_text(model, text, tokenizer, max_length=args.generate_max_length, greedy=args.greedy, top_k=args.top_k, top_p=args.top_p)
            print(output)
            texts = []

if __name__ == "__main__":
    main()
