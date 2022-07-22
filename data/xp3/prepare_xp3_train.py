from functools import partial
import multiprocessing
import os

import datasets
from datasets import load_dataset
# pip install -q iso-639
from iso639 import languages
from promptsource.templates import DatasetTemplates

# Set to False to use multilingual prompts e.g. 'id' for xcopa/id instead of 'en'
USE_ENGLISH_PROMPTS = True

DS_TO_ENG_PROMPT = {
    "xcopa": "en",
    "Muennighoff/xstory_cloze": "en",
    'GEM/wiki_lingua': 'en',
    'xnli': 'en',
    "paws-x": "en",
    "xquad": "en",
    "khalidalt/tydiqa-primary": "english",
    "khalidalt/tydiqa-goldp": "english",
}

BIAS_FAIRNESS = [
    ('crows_pairs', None),
    ('jigsaw_toxicity_pred', None),
    ('super_glue','axg'),
    ('wino_bias','type1_anti'),
    ('wino_bias','type2_anti'),
    ('wino_bias','type1_pro'),
    ('wino_bias','type2_pro'),
]

EVAL_DATASETS_L1 = [
    ('super_glue','wsc.fixed'),
    ('winogrande','winogrande_xl'),
    ('super_glue','cb'),
    ('super_glue','rte'),
    ('anli',None),  
    ('story_cloze', '2016'),
    ('Muennighoff/xstory_cloze', 'ar'),
    ('Muennighoff/xstory_cloze', 'es'),
    ('Muennighoff/xstory_cloze', 'eu'),
    ('Muennighoff/xstory_cloze', 'id'),
    ('Muennighoff/xstory_cloze', 'hi'),
    ('Muennighoff/xstory_cloze', 'te'),
    ('Muennighoff/xstory_cloze', 'sw'),
    ('Muennighoff/xstory_cloze', 'zh'),
    ('hellaswag', None),
    ('super_glue', 'copa'),
    ('super_glue', 'wic'),
    # Multilingual
    ('Muennighoff/xwinograd','en'),
    ('Muennighoff/xwinograd','fr'),
    ('Muennighoff/xwinograd','pt'),
    ('Muennighoff/xwinograd','zh'),
    ('xcopa','id'),
    ('xcopa','ta'),
    ('xcopa','sw'),
    ('xcopa','vi'),
    ('xcopa','zh'),
    ("xnli", "ar"),
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
]

EVAL_DATASETS_L2 = [
    ("xnli", "bg"),
]

TRAIN_DATASETS = [
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
    ('super_glue','multirc'),
    ('wiki_hop','original'),
    ('wiqa',None),
    ('piqa',None),
    ('amazon_polarity',None),
    ('app_reviews',None),
    ('imdb',None),
    ('rotten_tomatoes',None),
    ('yelp_review_full',None),
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
    ('khalidalt/tydiqa-primary', 'arabic'),
    ('khalidalt/tydiqa-primary', 'bengali'),
    ('khalidalt/tydiqa-primary', 'english'),
    ('khalidalt/tydiqa-primary', 'indonesian'),
    ('khalidalt/tydiqa-primary', 'swahili'),
    ('khalidalt/tydiqa-primary', 'telugu'),
    ('khalidalt/tydiqa-goldp', 'arabic'),
    ('khalidalt/tydiqa-goldp', 'bengali'),
    ('khalidalt/tydiqa-goldp', 'english'),
    ('khalidalt/tydiqa-goldp', 'indonesian'),
    ('khalidalt/tydiqa-goldp', 'swahili'),
    ('khalidalt/tydiqa-goldp', 'telugu'),
    ('Muennighoff/mbpp', 'sanitized'),
    ("openai_humaneval", None),
    # flores200
]

FLORES_LANGS = [
    ("Acehnese (Arabic script)", "ace_Arab"),
    ("Acehnese (Latin script)", "ace_Latn"),
    ("Mesopotamian Arabic", "acm_Arab"),
    ("Ta’izzi-Adeni Arabic", "acq_Arab"),
    ("Tunisian Arabic", "aeb_Arab"),
    ("Afrikaans", "afr_Latn"),
    ("South Levantine Arabic", "ajp_Arab"),
    ("Akan", "aka_Latn"),
    ("Amharic", "amh_Ethi"),
    ("North Levantine Arabic", "apc_Arab"),
    ("Modern Standard Arabic", "arb_Arab"),
    ("Modern Standard Arabic (Romanized)", "arb_Latn"),
    ("Najdi Arabic", "ars_Arab"),
    ("Moroccan Arabic", "ary_Arab"),
    ("Egyptian Arabic", "arz_Arab"),
    ("Assamese", "asm_Beng"),
    ("Asturian", "ast_Latn"),
    ("Awadhi", "awa_Deva"),
    ("Central Aymara", "ayr_Latn"),
    ("South Azerbaijani", "azb_Arab"),
    ("North Azerbaijani", "azj_Latn"),
    ("Bashkir", "bak_Cyrl"),
    ("Bambara", "bam_Latn"),
    ("Balinese", "ban_Latn"),
    ("Belarusian", "bel_Cyrl"),
    ("Bemba", "bem_Latn"),
    ("Bengali", "ben_Beng"),
    ("Bhojpuri", "bho_Deva"),
    ("Banjar (Arabic script)", "bjn_Arab"),
    ("Banjar (Latin script)", "bjn_Latn"),
    ("Standard Tibetan", "bod_Tibt"),
    ("Bosnian", "bos_Latn"),
    ("Buginese", "bug_Latn"),
    ("Bulgarian", "bul_Cyrl"),
    ("Catalan", "cat_Latn"),
    ("Cebuano", "ceb_Latn"),
    ("Czech", "ces_Latn"),
    ("Chokwe", "cjk_Latn"),
    ("Central Kurdish", "ckb_Arab"),
    ("Crimean Tatar", "crh_Latn"),
    ("Welsh", "cym_Latn"),
    ("Danish", "dan_Latn"),
    ("German", "deu_Latn"),
    ("Southwestern Dinka", "dik_Latn"),
    ("Dyula", "dyu_Latn"),
    ("Dzongkha", "dzo_Tibt"),
    ("Greek", "ell_Grek"),
    ("English", "eng_Latn"),
    ("Esperanto", "epo_Latn"),
    ("Estonian", "est_Latn"),
    ("Basque", "eus_Latn"),
    ("Ewe", "ewe_Latn"),
    ("Faroese", "fao_Latn"),
    ("Fijian", "fij_Latn"),
    ("Finnish", "fin_Latn"),
    ("Fon", "fon_Latn"),
    ("French", "fra_Latn"),
    ("Friulian", "fur_Latn"),
    ("Nigerian Fulfulde", "fuv_Latn"),
    ("Scottish Gaelic", "gla_Latn"),
    ("Irish", "gle_Latn"),
    ("Galician", "glg_Latn"),
    ("Guarani", "grn_Latn"),
    ("Gujarati", "guj_Gujr"),
    ("Haitian Creole", "hat_Latn"),
    ("Hausa", "hau_Latn"),
    ("Hebrew", "heb_Hebr"),
    ("Hindi", "hin_Deva"),
    ("Chhattisgarhi", "hne_Deva"),
    ("Croatian", "hrv_Latn"),
    ("Hungarian", "hun_Latn"),
    ("Armenian", "hye_Armn"),
    ("Igbo", "ibo_Latn"),
    ("Ilocano", "ilo_Latn"),
    ("Indonesian", "ind_Latn"),
    ("Icelandic", "isl_Latn"),
    ("Italian", "ita_Latn"),
    ("Javanese", "jav_Latn"),
    ("Japanese", "jpn_Jpan"),
    ("Kabyle", "kab_Latn"),
    ("Jingpho", "kac_Latn"),
    ("Kamba", "kam_Latn"),
    ("Kannada", "kan_Knda"),
    ("Kashmiri (Arabic script)", "kas_Arab"),
    ("Kashmiri (Devanagari script)", "kas_Deva"),
    ("Georgian", "kat_Geor"),
    ("Central Kanuri (Arabic script)", "knc_Arab"),
    ("Central Kanuri (Latin script)", "knc_Latn"),
    ("Kazakh", "kaz_Cyrl"),
    ("Kabiyè", "kbp_Latn"),
    ("Kabuverdianu", "kea_Latn"),
    ("Khmer", "khm_Khmr"),
    ("Kikuyu", "kik_Latn"),
    ("Kinyarwanda", "kin_Latn"),
    ("Kyrgyz", "kir_Cyrl"),
    ("Kimbundu", "kmb_Latn"),
    ("Northern Kurdish", "kmr_Latn"),
    ("Kikongo", "kon_Latn"),
    ("Korean", "kor_Hang"),
    ("Lao", "lao_Laoo"),
    ("Ligurian", "lij_Latn"),
    ("Limburgish", "lim_Latn"),
    ("Lingala", "lin_Latn"),
    ("Lithuanian", "lit_Latn"),
    ("Lombard", "lmo_Latn"),
    ("Latgalian", "ltg_Latn"),
    ("Luxembourgish", "ltz_Latn"),
    ("Luba-Kasai", "lua_Latn"),
    ("Ganda", "lug_Latn"),
    ("Luo", "luo_Latn"),
    ("Mizo", "lus_Latn"),
    ("Standard Latvian", "lvs_Latn"),
    ("Magahi", "mag_Deva"),
    ("Maithili", "mai_Deva"),
    ("Malayalam", "mal_Mlym"),
    ("Marathi", "mar_Deva"),
    ("Minangkabau (Arabic script)", "min_Arab"),
    ("Minangkabau (Latin script)", "min_Latn"),
    ("Macedonian", "mkd_Cyrl"),
    ("Plateau Malagasy", "plt_Latn"),
    ("Maltese", "mlt_Latn"),
    ("Meitei (Bengali script)", "mni_Beng"),
    ("Halh Mongolian", "khk_Cyrl"),
    ("Mossi", "mos_Latn"),
    ("Maori", "mri_Latn"),
    ("Burmese", "mya_Mymr"),
    ("Dutch", "nld_Latn"),
    ("Norwegian Nynorsk", "nno_Latn"),
    ("Norwegian Bokmål", "nob_Latn"),
    ("Nepali", "npi_Deva"),
    ("Northern Sotho", "nso_Latn"),
    ("Nuer", "nus_Latn"),
    ("Nyanja", "nya_Latn"),
    ("Occitan", "oci_Latn"),
    ("West Central Oromo", "gaz_Latn"),
    ("Odia", "ory_Orya"),
    ("Pangasinan", "pag_Latn"),
    ("Eastern Panjabi", "pan_Guru"),
    ("Papiamento", "pap_Latn"),
    ("Western Persian", "pes_Arab"),
    ("Polish", "pol_Latn"),
    ("Portuguese", "por_Latn"),
    ("Dari", "prs_Arab"),
    ("Southern Pashto", "pbt_Arab"),
    ("Ayacucho Quechua", "quy_Latn"),
    ("Romanian", "ron_Latn"),
    ("Rundi", "run_Latn"),
    ("Russian", "rus_Cyrl"),
    ("Sango", "sag_Latn"),
    ("Sanskrit", "san_Deva"),
    ("Santali", "sat_Olck"),
    ("Sicilian", "scn_Latn"),
    ("Shan", "shn_Mymr"),
    ("Sinhala", "sin_Sinh"),
    ("Slovak", "slk_Latn"),
    ("Slovenian", "slv_Latn"),
    ("Samoan", "smo_Latn"),
    ("Shona", "sna_Latn"),
    ("Sindhi", "snd_Arab"),
    ("Somali", "som_Latn"),
    ("Southern Sotho", "sot_Latn"),
    ("Spanish", "spa_Latn"),
    ("Tosk Albanian", "als_Latn"),
    ("Sardinian", "srd_Latn"),
    ("Serbian", "srp_Cyrl"),
    ("Swati", "ssw_Latn"),
    ("Sundanese", "sun_Latn"),
    ("Swedish", "swe_Latn"),
    ("Swahili", "swh_Latn"),
    ("Silesian", "szl_Latn"),
    ("Tamil", "tam_Taml"),
    ("Tatar", "tat_Cyrl"),
    ("Telugu", "tel_Telu"),
    ("Tajik", "tgk_Cyrl"),
    ("Tagalog", "tgl_Latn"),
    ("Thai", "tha_Thai"),
    ("Tigrinya", "tir_Ethi"),
    ("Tamasheq (Latin script)", "taq_Latn"),
    ("Tamasheq (Tifinagh script)", "taq_Tfng"),
    ("Tok Pisin", "tpi_Latn"),
    ("Tswana", "tsn_Latn"),
    ("Tsonga", "tso_Latn"),
    ("Turkmen", "tuk_Latn"),
    ("Tumbuka", "tum_Latn"),
    ("Turkish", "tur_Latn"),
    ("Twi", "twi_Latn"),
    ("Central Atlas Tamazight", "tzm_Tfng"),
    ("Uyghur", "uig_Arab"),
    ("Ukrainian", "ukr_Cyrl"),
    ("Umbundu", "umb_Latn"),
    ("Urdu", "urd_Arab"),
    ("Northern Uzbek", "uzn_Latn"),
    ("Venetian", "vec_Latn"),
    ("Vietnamese", "vie_Latn"),
    ("Waray", "war_Latn"),
    ("Wolof", "wol_Latn"),
    ("Xhosa", "xho_Latn"),
    ("Eastern Yiddish", "ydd_Hebr"),
    ("Yoruba", "yor_Latn"),
    ("Yue Chinese", "yue_Hant"),
    ("Chinese (Simplified)", "zho_Hans"),
    ("Chinese (Traditional)", "zho_Hant"),
    ("Standard Malay", "zsm_Latn"),
    ("Zulu", "zul_Latn"),
]

# Copied from metadata
BLOOM_LANGS = """
- ak
- ar
- as
- bm
- bn
- ca
- code
- en
- es
- eu
- fon
- fr
- gu
- hi
- id
- ig
- ki
- kn
- lg
- ln
- ml
- mr
- ne
- nso
- ny
- or
- pa
- pt
- rn
- rw
- sn
- st
- sw
- ta
- te
- tn
- ts
- tum
- tw
- ur
- vi
- wo
- xh
- yo
- zh
- zu
"""

DS_TO_LANG = {
    'Muennighoff/mbpp': 'code',
    'openai_humaneval': 'code',
    "cmn": "zh", # == zho
    "npi": "ne", # == npe
    "ory": "or", # == ori
    "swh": "sw", # == swa
}

bloom_lang_codes_iso3 = []
bloom_lang_codes_iso2 = []
for lang in BLOOM_LANGS.split("\n")[1:-1]:
    iso2 = lang.replace("- ", "")
    DS_TO_LANG[iso2] = iso2
    try:
        name = languages.get(alpha2=iso2)
        DS_TO_LANG[name.name.lower()] = iso2
        # name is e.g. 'swahili (macrolanguage)' also add swahili
        DS_TO_LANG[name.name.lower().split(" ")[0]] = iso2

        iso3 = name.part3
        DS_TO_LANG[iso3] = iso2
    except KeyError:
        print(f"Could not find iso3 code for {lang}.")

for (l1_name, l1_code) in FLORES_LANGS:
    for (l2_name, l2_code) in FLORES_LANGS:
        if l1_code.split("_")[0] not in DS_TO_LANG or l2_code.split("_")[0] not in DS_TO_LANG:
            print(f"Skipping as {l1_name} or {l2_name} was not pre-trained on.")
            continue
        elif l1_name == l2_name:
            continue
        TRAIN_DATASETS.append(("facebook/flores", f"{l1_code}-{l2_code}"))


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

# Adapted from t0.seqio_tasks.utils
def apply_template(dataset, template):
    def map_fn(ex):
        ex = removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            # TODO: Use str to convert e.g. "\u00e9" -> é ?
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            # inputs is a str by default & targets a list
            ex = {"inputs": "", "targets": []}

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
    
    lang_dir = DS_TO_LANG.get(ds_name, None)
    if lang_dir is None:
        lang_dir = DS_TO_LANG.get(subset_name, "en")
    if ds_name == "facebook/flores":
         lang_dir = DS_TO_LANG.get(subset_name.split("-")[-1].split("_")[0])

    os.makedirs(lang_dir, exist_ok=True)

    ds = load_dataset(ds_name, subset_name)

    dataset_splits = list(ds.keys())

    if split == "validation":
        if split not in dataset_splits or len(dataset_splits) > 1:
            print(f"Validation not found for {ds_name}")
            return
        dataset_splits = ["validation"]
    elif split == "train":
        #dataset_splits = [sp for sp in dataset_splits if sp in ["train", "validation", "test"]]
        # Use as much as possible
        # Will need to remove e.g. test datasets to benchmark same task performance
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
            # TODO: Count tokens / language
            if ds_name == "tatoeba_mt":
                # E.g. translate-this-ara-eng, where eng is the target
                lang_dir = t_name.split("-")[-1]

            out_path = os.path.join(
                lang_dir, 
                f'xp3_{ds_name}_{subset_name}_{split}_{t_name}.jsonl'.replace("/", "_").replace(" ", "_")
            )
            if os.path.exists(out_path) and ds_name not in ['Muennighoff/mbpp']:
                print("SKIPPING AS EXISTS:", out_path)
                continue
            # Do not force ascii to allow chars like é
            apply_template(dataset=ds[split], template=prompts[t_name]).to_json(out_path,  orient="records", lines=True, force_ascii=False)


# Testing:
#TRAIN_DATASETS = [
#    ("facebook/flores", "yor_Latn-zul_Latn"),
#]
#for ds in TRAIN_DATASETS:
#    write_to_jsonl_hub(ds)


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(partial(write_to_jsonl_hub, split="train"), TRAIN_DATASETS)
#    pool.map(partial(write_to_jsonl_hub, split="validation"), TRAIN_DATASETS)
