import argparse
import random

from promptsource.templates import DatasetTemplates


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--template_config_name",
        type=str,
        default=None,
        help="The name of the dataset_config_name of the template we want to use, example: use XNLI En prompts for XNLI Fr",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="templates.txt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--choices",
        type=int,
        default=5,
    )


    args = parser.parse_args()

    # TODO @thomasw21 hack!
    if args.dataset_config_name == "None":
        args.dataset_config_name = None
    if args.template_config_name == "None":
        args.template_config_name = None

    return args

def main():
    args = parse_args()

    random.seed(args.seed)
    
    if (args.dataset_config_name is None and args.template_config_name is None) or args.dataset_name == "anli":
        prompt_dataset_name = f"{args.dataset_name}"
    elif args.template_config_name is not None:
        prompt_dataset_name = f"{args.dataset_name}/{args.template_config_name}"
    else:
        prompt_dataset_name = f"{args.dataset_name}/{args.dataset_config_name}"

    prompts = DatasetTemplates(
        prompt_dataset_name
    )

    template_names = prompts.all_template_names

    with open(args.outfile, "a") as f:
        for choice in random.sample(population=template_names, k=min(len(template_names), args.choices)):
            f.write(f'{args.dataset_name},{args.dataset_config_name},{args.template_config_name},"{choice}"\n')

if __name__ == "__main__":
    main()
