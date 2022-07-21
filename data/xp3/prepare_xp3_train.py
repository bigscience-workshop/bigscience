from functools import partial
import multiprocessing

import datasets
from datasets import load_dataset

from promptsource.templates import DatasetTemplates


USE_ENGLISH_PROMPTS = True

DS_TO_ENG_PROMPT = {
    "xcopa": "en",
    'GEM/wiki_lingua': 'en',
    "paws-x": "en",
    "xquad": "en",
}

TRAIN_DATASETS = [
    ('crows_pairs', None),
    ('jigsaw_toxicity_pred', None),
    ('super_glue','axg'),
    ('wino_bias','type1_anti'),
    ('wino_bias','type2_anti'),
    ('wino_bias','type1_pro'),
    ('wino_bias','type2_pro'),
    ('super_glue','wsc.fixed'),
    # ('winogrande','winogrande_xl'),
    ('super_glue','cb'),
    ('super_glue','rte'),
    ('anli',None),
    ('glue','mrpc'),
    ('glue','qqp'),
    ('paws','labeled_final'),
    ('ai2_arc','ARC-Challenge'),
    ('ai2_arc','ARC-Easy'),
    ('kilt_tasks','hotpotqa'),
    ('trivia_qa','unfiltered'),
    ('web_questions',None),
    ('wiki_qa',None),
    ('adversarial_qa','dbidaf'),
    ('adversarial_qa','dbert'),
    ('adversarial_qa','droberta'),
    ('duorc','SelfRC'),
    ('duorc','ParaphraseRC'),
    ('ropes',None),
    ('squad_v2',None),
    ('super_glue','record'),
    ('quoref',None),
    ('cos_e','v1.11'),
    ('cosmos_qa',None),
    ('dream',None),
    ('openbookqa','main'),
    ('qasc',None),
    ('quail',None),
    ('quarel',None),
    ('quartz',None),
    ('race','high'),
    ('race','middle'),
    ('sciq',None),
    ('social_i_qa',None),
    ('super_glue','boolq'),
    ('super_glue','copa'),
    ('super_glue','multirc'),
    ('wiki_hop','original'),
    ('wiqa',None),
    ('piqa',None),
    ('amazon_polarity',None),
    ('app_reviews',None),
    ('imdb',None),
    ('rotten_tomatoes',None),
    ('yelp_review_full',None),
    ('story_cloze','2016'),
    ('hellaswag',None),
    ('common_gen',None),
    ('wiki_bio',None),
    ('cnn_dailymail','3.0.0'),
    ('gigaword',None),
    ('multi_news',None),
    ('samsum',None),
    ('xsum',None),
    ('ag_news',None),
    ('dbpedia_14',None),
    ('trec',None),
    ('super_glue','wic'),
    # Multilingual
    ('GEM/wiki_lingua', 'ar'),
    ('GEM/wiki_lingua', 'en'),
    ('GEM/wiki_lingua', 'es'),
    ('GEM/wiki_lingua', 'fr'),
    ('GEM/wiki_lingua', 'hi'),
    ('GEM/wiki_lingua', 'id'),
    ('GEM/wiki_lingua', 'pt'),
    ('GEM/wiki_lingua', 'vi'),
    ('GEM/wiki_lingua', 'zh'),
    # ('Muennighoff/xwinograd','en'),
    # ('Muennighoff/xwinograd','fr'),
    # ('Muennighoff/xwinograd','pt'),
    # ('Muennighoff/xwinograd','zh'),
    # ('xcopa','id'),
    # ('xcopa','ta'),
    # ('xcopa','sw'),
    # ('xcopa','vi'),
    # ('xcopa','zh'),
    # ("xnli", "ar"),
    # ("xnli", "bg"),
    # ("xnli", "de"),
    # ("xnli", "el"),
    # ("xnli", "en"),
    # ("xnli", "es"),
    # ("xnli", "fr"),
    # ("xnli", "hi"),
    # ("xnli", "ru"),
    # ("xnli", "sw"),
    # ("xnli", "th"),
    # ("xnli", "tr"),
    # ("xnli", "ur"),
    # ("xnli", "vi"),
    # ("xnli", "zh"),
    ('tatoeba_mt', 'ara-eng'),
    ('tatoeba_mt', 'ara-fra'),
    ('tatoeba_mt', 'ara-spa'),
    ('tatoeba_mt', 'ben-eng'),
    ('tatoeba_mt', 'cat-eng'),
    ('tatoeba_mt', 'cat-fra'),
    ('tatoeba_mt', 'cat-por'),
    ('tatoeba_mt', 'cat-spa'),
    ('tatoeba_mt', 'eng-cmn_Hans'),
    ('tatoeba_mt', 'eng-cmn_Hant'),
    ('tatoeba_mt', 'eng-eus'),
    ('tatoeba_mt', 'eng-fra'),
    ('tatoeba_mt', 'eng-hin'),
    ('tatoeba_mt', 'eng-ind'),
    ('tatoeba_mt', 'eng-mal'),
    ('tatoeba_mt', 'eng-mar'),
    ('tatoeba_mt', 'eng-por'),
    ('tatoeba_mt', 'eng-run'),
    ('tatoeba_mt', 'eng-spa'),
    ('tatoeba_mt', 'eng-swa'),
    ('tatoeba_mt', 'eng-tam'),
    ('tatoeba_mt', 'eng-tel'),
    ('tatoeba_mt', 'eng-urd'),
    ('tatoeba_mt', 'eng-vie'),
    ('tatoeba_mt', 'eng-zho'),
    ('tatoeba_mt', 'eus-spa'),
    ('tatoeba_mt', 'fra-cmn_Hans'),
    ('tatoeba_mt', 'fra-cmn_Hant'),
    ('tatoeba_mt', 'fra-ind'),
    ('tatoeba_mt', 'fra-por'),
    ('tatoeba_mt', 'fra-run'),
    ('tatoeba_mt', 'fra-spa'),
    ('tatoeba_mt', 'fra-vie'),
    ('tatoeba_mt', 'fra-zho'),
    ('tatoeba_mt', 'hin-urd'),
    ('tatoeba_mt', 'hin-zho'),
    ('tatoeba_mt', 'por-cmn_Hans'),
    ('tatoeba_mt', 'por-cmn_Hant'),
    ('tatoeba_mt', 'por-spa'),
    ('tatoeba_mt', 'por-zho'),
    ('tatoeba_mt', 'run-spa'),
    ('tatoeba_mt', 'spa-cmn_Hans'),
    ('tatoeba_mt', 'spa-cmn_Hant'),
    ('tatoeba_mt', 'spa-vie'),
    ('tatoeba_mt', 'spa-zho'),
    ('tatoeba_mt', 'vie-cmn_Hans'),
    ('tatoeba_mt', 'vie-zho'),
    ('xquad.ar', None),
    ('xquad.zh', None),
    ('xquad.vi', None),
    ('xquad.en', None),
    ('xquad.es', None),
    ('xquad.hi', None),
    ('paws-x', 'en'),
    ('paws-x', 'es'),
    ('paws-x', 'fr'),
    ('paws-x', 'zh'),
    # ('tydiqa', 'primary_task'),
    # ('tydiqa', 'secondary_task'),
    ('Muennighoff/mbpp', 'sanitized'),
    ("openai_humaneval", None),
]

TRAIN_DATASETS = [
    ('GEM/wiki_lingua', 'en'),
    ('xquad', 'vi'),
    ('xquad', 'en'),
    ('xquad', 'es'),
    ('xquad', 'hi'),
    ('paws-x', 'en'),
    ('paws-x', 'es'),
]

DS_TO_LANG = {
    'Muennighoff/mbpp': 'code',
    'openai_humaneval': 'code',
}

# Copied from promptsource.utils
def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example

# Copied from t0.seqio_tasks.utils
def apply_template(dataset, template):
    def map_fn(ex):
        ex = removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})

# Copied from t0.seqio_tasks.utils
def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits

def write_to_jsonl_hub(ds, split="train"):
    ds_name, subset_name = ds

    ds = load_dataset(ds_name, subset_name)

    dataset_splits = list(ds.keys())

    if split == "validation":
        if split not in dataset_splits or len(dataset_splits) > 1:
            print(f"Validation not found for {ds_name}")
            return
        dataset_splits = ["validation"]
    elif split == "train":
        dataset_splits = [sp for sp in dataset_splits if sp in ["train", "validation", "test"]]
        if len(dataset_splits) > 1 and "validation" in dataset_splits:
            dataset_splits.remove("validation")
    

    #dataset_splits = {x:x for x in ds.keys()}
    #get_dataset_splits(ds_name, subset_name)
    # Always use >=1 ds for training
    #if split == "train" and len(dataset_splits) > 1:
    #    dataset_splits.pop("validation")
    #    dataset_splits = 
    #elif split == "validation":
    #    dataset_splits.pop("train")
    #    dataset_splits.pop("test")
    #    assert len(dataset_splits) in [0,1]
    #    if len(dataset_splits) == 0:
    #        return
    #ds = load_dataset(ds_name, subset_name)

    subset_name_prompt = subset_name
    if USE_ENGLISH_PROMPTS and ds_name in DS_TO_ENG_PROMPT:
        subset_name_prompt = DS_TO_ENG_PROMPT[ds_name]

    prompts = DatasetTemplates(f"{ds_name}/{subset_name_prompt}")
    # TODO: Add capping? (cap = MAX_EXAMPLES_PER_DATASET // num_templates)
    for split in dataset_splits:
        for t_name in prompts.all_template_names:
            # TODO: Add language information
            # TODO: Count tokens / language
            out_path = f'xp3_{ds_name}_{subset_name}_{split}_{t_name}.jsonl'.replace("/", "_")
            apply_template(dataset=ds[split], template=prompts[t_name]).to_json(out_path)

for ds in TRAIN_DATASETS:
    write_to_jsonl_hub(ds)


#with multiprocessing.Pool(processes=1) as pool:#multiprocessing.cpu_count()
#    pool.map(partial(write_to_jsonl_hub, split="train"), TRAIN_DATASETS)
#    pool.map(partial(write_to_jsonl_hub, split="validation"), TRAIN_DATASETS)
