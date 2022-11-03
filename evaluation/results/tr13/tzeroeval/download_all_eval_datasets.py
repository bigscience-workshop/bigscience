from datasets import load_dataset

# (dataset_name, dataset_config)
T0_EVAL_DATASETS=[
    ("super_glue", "copa"),
    ("anli", None),
    ("super_glue", "cb"),
    ("super_glue", "rte"),
    ("super_glue", "wsc.fixed"),
    ("winogrande", "winogrande_xl"),
    ("super_glue", "wic"),
    ("hellaswag", None),
    # TODO @thomasw21 this is a manual download
    # ("story_cloze_2016": ('StoryCloze', 0.5),
]

MULTILINGUAL_DATASETS=[
    ('Muennighoff/xwinograd','en'),
    ('Muennighoff/xwinograd','fr'),
    ('Muennighoff/xwinograd','pt'),
    ('Muennighoff/xwinograd','zh'),
    ('Muennighoff/xwinograd','jp'),
    ('Muennighoff/xwinograd','ru'),
    ('xcopa','id'),
    ('xcopa','ta'),
    ('xcopa','sw'),
    ('xcopa','vi'),
    ('xcopa','zh'),
    ('xcopa','tr'),
    ('xcopa','qu'),
    ('xcopa','it'),
    ('xcopa','ht'),
    ('xcopa','et'),
    ("xnli", "ar"),
    ("xnli", "bg"),
    ("xnli", "de"),
    ("xnli", "el"),
    ("xnli", "en"),
    ("xnli", "es"),
    ("xnli", "fr"),
    ("xnli", "hi"),
    ("xnli", "ru"),
    ("xnli", "sw"),
    ("xnli", "th"),
    ("xnli", "tr"),
    ("xnli", "ur"),
    ("xnli", "vi"),
    ("xnli", "zh"),
    ("xnli", "all_languages"),
]

def main():
    for dataset_name, dataset_config in T0_EVAL_DATASETS + MULTILINGUAL_DATASETS:
        load_dataset(dataset_name, dataset_config, split="validation")

if __name__ == "__main__":
    main()
