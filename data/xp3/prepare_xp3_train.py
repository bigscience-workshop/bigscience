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
    "mlqa": "mlqa.en.en",
    "xquad": "xquad.en",
    "khalidalt/tydiqa-primary": "english",
    "khalidalt/tydiqa-goldp": "english",
    "pasinit/xlwic": "en",
}

TRUNCATE = {
    "khalidalt/tydiqa-primary": {
        "cols": ["document_plaintext", "question_text"], 
        "trunc_col": "document_plaintext"
        },
    "GEM/wiki_lingua": {
        "cols": ["document_plaintext", "question_text"], 
        "trunc_col": "document_plaintext"
        },
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
    # Multilingual
    ('Muennighoff/xwinograd','en'),
    ('Muennighoff/xwinograd','fr'),
    ('Muennighoff/xwinograd','pt'),
    ('Muennighoff/xwinograd','zh'),
    ('clue', 'cluewsc2020'),
    ('xcopa','id'),
    ('xcopa','ta'),
    ('xcopa','sw'),
    ('xcopa','vi'),
    ('xcopa','zh'),
    ("xnli", "ar"),
    ("xnli", "en"),
    ("xnli", "es"),
    ("xnli", "fr"),
    ("xnli", "hi"),
    ("xnli", "sw"),
    ("xnli", "ur"),
    ("xnli", "vi"),
    ("xnli", "zh"),
]

EVAL_DATASETS_L2 = [
    ('Muennighoff/xwinograd','jp'),
    ('Muennighoff/xwinograd','ru'),
    ('xcopa','et'),
    ('xcopa','ht'),
    ('xcopa','it'),
    ('xcopa','qu'),
    ('xcopa','th'),
    ('xcopa','tr'),
    ("xnli", "bg"),
    ("xnli", "de"),
    ("xnli", "el"),
    ("xnli", "ru"),
    ("xnli", "th"),
    ("xnli", "tr"),
]

TRAIN_DATASETS = [
    # English-only
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
    ('Helsinki-NLP/tatoeba_mt', 'ara-eng'),
    ('Helsinki-NLP/tatoeba_mt', 'ara-fra'),
    ('Helsinki-NLP/tatoeba_mt', 'ara-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'ben-eng'),
    ('Helsinki-NLP/tatoeba_mt', 'cat-eng'),
    ('Helsinki-NLP/tatoeba_mt', 'cat-fra'),
    ('Helsinki-NLP/tatoeba_mt', 'cat-por'),
    ('Helsinki-NLP/tatoeba_mt', 'cat-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-cmn_Hans'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-cmn_Hant'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-eus'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-fra'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-hin'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-ind'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-mal'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-mar'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-por'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-run'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-swa'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-tam'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-tel'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-urd'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-vie'),
    ('Helsinki-NLP/tatoeba_mt', 'eng-zho'),
    ('Helsinki-NLP/tatoeba_mt', 'eus-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-cmn_Hans'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-cmn_Hant'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-ind'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-por'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-run'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-vie'),
    ('Helsinki-NLP/tatoeba_mt', 'fra-zho'),
    ('Helsinki-NLP/tatoeba_mt', 'hin-urd'),
    ('Helsinki-NLP/tatoeba_mt', 'hin-zho'),
    ('Helsinki-NLP/tatoeba_mt', 'por-cmn_Hans'),
    ('Helsinki-NLP/tatoeba_mt', 'por-cmn_Hant'),
    ('Helsinki-NLP/tatoeba_mt', 'por-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'por-zho'),
    ('Helsinki-NLP/tatoeba_mt', 'run-spa'),
    ('Helsinki-NLP/tatoeba_mt', 'spa-cmn_Hans'),
    ('Helsinki-NLP/tatoeba_mt', 'spa-cmn_Hant'),
    ('Helsinki-NLP/tatoeba_mt', 'spa-vie'),
    ('Helsinki-NLP/tatoeba_mt', 'spa-zho'),
    ('Helsinki-NLP/tatoeba_mt', 'vie-cmn_Hans'),
    ('Helsinki-NLP/tatoeba_mt', 'vie-zho'),
    ('xquad', 'xquad.ar'),
    ('xquad', 'xquad.zh'),
    ('xquad', 'xquad.vi'),
    ('xquad', 'xquad.en'),
    ('xquad', 'xquad.es'),
    ('xquad', 'xquad.hi'),
    ('mlqa', 'mlqa.ar.ar'),
    ('mlqa', 'mlqa.vi.vi'),
    ('mlqa', 'mlqa.zh.zh'),
    ('mlqa', 'mlqa.es.es'),
    ('mlqa', 'mlqa.en.en'),
    ('mlqa', 'mlqa.hi.hi'),

    ('mlqa', 'mlqa.ar.vi'),
    ('mlqa', 'mlqa.ar.zh'),
    ('mlqa', 'mlqa.ar.es'),
    ('mlqa', 'mlqa.ar.en'),
    ('mlqa', 'mlqa.ar.hi'),

    ('mlqa', 'mlqa.vi.ar'),
    ('mlqa', 'mlqa.vi.zh'),
    ('mlqa', 'mlqa.vi.es'),
    ('mlqa', 'mlqa.vi.en'),
    ('mlqa', 'mlqa.vi.hi'),

    ('mlqa', 'mlqa.zh.ar'),
    ('mlqa', 'mlqa.zh.vi'),
    ('mlqa', 'mlqa.zh.es'),
    ('mlqa', 'mlqa.zh.en'),
    ('mlqa', 'mlqa.zh.hi'),

    ('mlqa', 'mlqa.es.ar'),
    ('mlqa', 'mlqa.es.vi'),
    ('mlqa', 'mlqa.es.zh'),
    ('mlqa', 'mlqa.es.en'),
    ('mlqa', 'mlqa.es.hi'),

    ('mlqa', 'mlqa.en.ar'),
    ('mlqa', 'mlqa.es.vi'),
    ('mlqa', 'mlqa.es.zh'),
    ('mlqa', 'mlqa.es.es'),
    ('mlqa', 'mlqa.es.hi'),

    ('mlqa', 'mlqa.hi.ar'),
    ('mlqa', 'mlqa.hi.vi'),
    ('mlqa', 'mlqa.hi.zh'),
    ('mlqa', 'mlqa.hi.es'),
    ('mlqa', 'mlqa.hi.en'),

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
    ("great_code", None),
    ("neural_code_search", "evaluation_dataset"),
    ("codeparrot/codecomplex", "codeparrot--codecomplex"),
    ('clue', 'c3'),
    ('clue', 'cmrc2018'),
    ('clue', 'csl'),
    ('clue', 'drcd'),
    ('clue', 'tnews'),
    ('super_glue', 'wic'),
    ('pasinit/xlwic', "xlwic_en_zh"),
    ('pasinit/xlwic', "xlwic_fr_fr"),
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

WMT22_LANGS = [
    ("afr", "eng"),
    ("afr", "som"),
    ("amh", "eng"),
    ("amh", "fra"),
    ("amh", "nya"),
    ("amh", "orm"),
    ("amh", "sna"),
    ("amh", "som"),
    ("amh", "ssw"),
    ("amh", "swh"),
    ("amh", "tsn"),
    ("amh", "tso"),
    ("amh", "umb"),
    ("amh", "xho"),
    ("amh", "yor"),
    ("amh", "zul"),
    ("eng", "fuv"),
    ("eng", "hau"),
    ("eng", "ibo"),
    ("eng", "kam"),
    ("eng", "kin"),
    ("eng", "lin"),
    ("eng", "lug"),
    ("eng", "luo"),
    ("eng", "nso"),
    ("eng", "nya"),
    ("eng", "orm"),
    ("eng", "sna"),
    ("eng", "som"),
    ("eng", "ssw"),
    ("eng", "swh"),
    ("eng", "tsn"),
    ("eng", "tso"),
    ("eng", "umb"),
    ("eng", "wol"),
    ("eng", "xho"),
    ("eng", "yor"),
    ("eng", "zul"),
    ("fra", "hau"),
    ("fra", "ibo"),
    ("fra", "kam"),
    ("fra", "kin"),
    ("fra", "lin"),
    ("fra", "lug"),
    ("fra", "luo"),
    ("fra", "nso"),
    ("fra", "nya"),
    ("fra", "orm"),
    ("fra", "som"),
    ("fra", "ssw"),
    ("fra", "swh"),
    ("fra", "tsn"),
    ("fra", "tso"),
    ("fra", "umb"),
    ("fra", "wol"),
    ("fra", "xho"),
    ("fra", "zul"),
    ("fuv", "hau"),
    ("fuv", "ibo"),
    ("fuv", "kam"),
    ("fuv", "kin"),
    ("fuv", "lug"),
    ("fuv", "luo"),
    ("fuv", "nso"),
    ("fuv", "nya"),
    ("fuv", "orm"),
    ("fuv", "sna"),
    ("fuv", "som"),
    ("fuv", "ssw"),
    ("fuv", "swh"),
    ("fuv", "tsn"),
    ("fuv", "tso"),
    ("fuv", "umb"),
    ("fuv", "xho"),
    ("fuv", "yor"),
    ("fuv", "zul"),
    ("hau", "ibo"),
    ("hau", "kam"),
    ("hau", "kin"),
    ("hau", "lug"),
    ("hau", "luo"),
    ("hau", "nso"),
    ("hau", "nya"),
    ("hau", "orm"),
    ("hau", "sna"),
    ("hau", "som"),
    ("hau", "ssw"),
    ("hau", "swh"),
    ("hau", "tsn"),
    ("hau", "tso"),
    ("hau", "umb"),
    ("hau", "xho"),
    ("hau", "yor"),
    ("hau", "zul"),
    ("ibo", "kam"),
    ("ibo", "kin"),
    ("ibo", "lug"),
    ("ibo", "luo"),
    ("ibo", "nso"),
    ("ibo", "nya"),
    ("ibo", "orm"),
    ("ibo", "sna"),
    ("ibo", "som"),
    ("ibo", "ssw"),
    ("ibo", "swh"),
    ("ibo", "tsn"),
    ("ibo", "tso"),
    ("ibo", "umb"),
    ("ibo", "xho"),
    ("ibo", "yor"),
    ("ibo", "zul"),
    ("kam", "kin"),
    ("kam", "lug"),
    ("kam", "luo"),
    ("kam", "nso"),
    ("kam", "nya"),
    ("kam", "orm"),
    ("kam", "sna"),
    ("kam", "som"),
    ("kam", "ssw"),
    ("kam", "swh"),
    ("kam", "tsn"),
    ("kam", "tso"),
    ("kam", "umb"),
    ("kam", "xho"),
    ("kam", "yor"),
    ("kam", "zul"),
    ("kin", "lug"),
    ("kin", "luo"),
    ("kin", "nso"),
    ("kin", "nya"),
    ("kin", "orm"),
    ("kin", "sna"),
    ("kin", "som"),
    ("kin", "ssw"),
    ("kin", "swh"),
    ("kin", "tsn"),
    ("kin", "tso"),
    ("kin", "umb"),
    ("kin", "xho"),
    ("kin", "yor"),
    ("kin", "zul"),
    ("lug", "luo"),
    ("lug", "nso"),
    ("lug", "nya"),
    ("lug", "orm"),
    ("lug", "sna"),
    ("lug", "som"),
    ("lug", "ssw"),
    ("lug", "swh"),
    ("lug", "tsn"),
    ("lug", "tso"),
    ("lug", "umb"),
    ("lug", "xho"),
    ("lug", "yor"),
    ("lug", "zul"),
    ("luo", "nso"),
    ("luo", "nya"),
    ("luo", "orm"),
    ("luo", "sna"),
    ("luo", "som"),
    ("luo", "ssw"),
    ("luo", "swh"),
    ("luo", "tsn"),
    ("luo", "tso"),
    ("luo", "umb"),
    ("luo", "xho"),
    ("luo", "yor"),
    ("luo", "zul"),
    ("nso", "nya"),
    ("nso", "orm"),
    ("nso", "sna"),
    ("nso", "som"),
    ("nso", "ssw"),
    ("nso", "swh"),
    ("nso", "tsn"),
    ("nso", "tso"),
    ("nso", "umb"),
    ("nso", "xho"),
    ("nso", "yor"),
    ("nso", "zul"),
    ("nya", "orm"),
    ("nya", "sna"),
    ("nya", "som"),
    ("nya", "ssw"),
    ("nya", "swh"),
    ("nya", "tsn"),
    ("nya", "tso"),
    ("nya", "umb"),
    ("nya", "xho"),
    ("nya", "yor"),
    ("nya", "zul"),
    ("orm", "sna"),
    ("orm", "som"),
    ("orm", "ssw"),
    ("orm", "swh"),
    ("orm", "tsn"),
    ("orm", "tso"),
    ("orm", "umb"),
    ("orm", "xho"),
    ("orm", "yor"),
    ("orm", "zul"),
    ("sna", "som"),
    ("sna", "ssw"),
    ("sna", "swh"),
    ("sna", "tsn"),
    ("sna", "tso"),
    ("sna", "umb"),
    ("sna", "xho"),
    ("sna", "yor"),
    ("sna", "zul"),
    ("som", "ssw"),
    ("som", "swh"),
    ("som", "tsn"),
    ("som", "tso"),
    ("som", "umb"),
    ("som", "wol"),
    ("som", "xho"),
    ("som", "yor"),
    ("som", "zul"),
    ("ssw", "swh"),
    ("ssw", "tsn"),
    ("ssw", "tso"),
    ("ssw", "umb"),
    ("ssw", "xho"),
    ("ssw", "yor"),
    ("ssw", "zul"),
    ("swh", "tsn"),
    ("swh", "tso"),
    ("swh", "umb"),
    ("swh", "xho"),
    ("swh", "yor"),
    ("swh", "zul"),
    ("tsn", "tso"),
    ("tsn", "umb"),
    ("tsn", "xho"),
    ("tsn", "yor"),
    ("tsn", "zul"),
    ("tso", "umb"),
    ("tso", "xho"),
    ("tso", "yor"),
    ("tso", "zul"),
    ("umb", "xho"),
    ("umb", "yor"),
    ("umb", "zul"),
    ("xho", "yor"),
    ("xho", "zul"),
    ("yor", "zul"),
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
    "great_code": "code",
    "neural_code_search": "code",
    "codeparrot/codecomplex": "code",
    "clue": "zh",
    "cmn": "zh", # == zho
    "npi": "ne", # == npe
    "ory": "or", # == ori
    "swh": "sw", # == swa
}

# Add GEM multilingual
WIKILINGUA_LANGS = ["ar", "en", "es", "fr", "hi", "id", "pt", "vi", "zh"]
for l1_code in WIKILINGUA_LANGS:
    for l2_code in WIKILINGUA_LANGS:
        if l1_code == l2_code:
            continue
        TRAIN_DATASETS.append(("GEM/wiki_lingua", f"{l1_code}_{l2_code}"))
        


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

TRAIN_DATASETS = []
for (l1_code, l2_code) in WMT22_LANGS:
    if l1_code not in DS_TO_LANG or l2_code not in DS_TO_LANG:
        print(f"Skipping as {l1_code} or {l2_code} was not pre-trained on.")
        continue
    elif l1_code == l2_code:
        continue
    TRAIN_DATASETS.append(("allenai/wmt22_african", f"{l1_code}-{l2_code}"))


### DATASET CREATION ###


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

#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom350m")

#import re
# Adapted from t0.seqio_tasks.utils
def apply_template(dataset, template, truncate_ds_name=None):
    #if truncate_ds_name is not None:
    #    #TRUNCATE
    #    template_toks = tokenizer.tokenize(re.sub("[\{].*?[\}]|\||\}", "", template.jinja))

    def map_fn(ex):
        ex = removeHyphen(ex)
        #if truncate_ds_name is not None:
        #    toks = sum([tokenizer.tokenize(ex[col]) for col in TRUNCATE[truncate_ds_name]["cols"]])
        #    if toks
        #    cols_toks = tokenizer.tokenize(ex)
        #    # col_to_truncate
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
            # inputs is a str by default & targets a list
            # Note that the signature changed in promptsource 
            # In 0.1.0 template.apply returned two strings; In >0.3.0 it retuns a str & list
            ex = {"inputs": "", "targets": [""]}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"][0]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})

# Copied from t0.seqio_tasks.utils
def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits

def add_language_name_wikilingua(example):
    example["source_language_name"] = languages.get(alpha2=example["source_language"]).name
    example["target_language_name"] = languages.get(alpha2=example["target_language"]).name
    return example

def filter_l1_l2_wikilingua(example, l1, l2):
    return example["source_language"] == l1 and example["target_language"] == l2


def write_to_jsonl_hub(ds, split="train"):
    ds_name, subset_name = ds

    is_wikilingua_cross_lingual = (ds_name == "GEM/wiki_lingua") and ("_") in subset_name
    
    lang_dir = DS_TO_LANG.get(ds_name, None)
    if lang_dir is None:
        lang_dir = DS_TO_LANG.get(subset_name, "en")
    if ds_name == "facebook/flores":
        lang_dir = DS_TO_LANG.get(subset_name.split("-")[-1].split("_")[0])
    elif is_wikilingua_cross_lingual:
        lang_dir = DS_TO_LANG.get(subset_name.split("_")[-1])
    elif ds_name == "xquad":
        lang_dir = DS_TO_LANG.get(subset_name.split(".")[1])
    elif ds_name == "mlqa":
        # Classify it by the target language for cross-lingual (i.e. what the loss is computed on)
        lang_dir = DS_TO_LANG.get(subset_name.split(".")[1])
    os.makedirs(lang_dir, exist_ok=True)

    if ds_name == "Helsinki-NLP/tatoeba_mt":
        # Fixes a bug when loading a ds where only test split exists
        ds = datasets.load_dataset(ds_name, subset_name, ignore_verifications=True, revision="842eb26634a9775f504bb2f3f43cd4cc5f9314d8")#, download_config=datasets.DownloadConfig(num_proc=1)
    else:
        ds = load_dataset(ds_name, subset_name)#, download_config=datasets.DownloadConfig(num_proc=1))

    # Filter down to only the current set
    if is_wikilingua_cross_lingual:
        # Keep only L1 -> L2 (L2 -> L1 will be a separate dataset)
        ds = ds.filter(partial(filter_l1_l2_wikilingua, l1=subset_name.split("_")[0], l2=subset_name.split("_")[1]))
        # Add names, e.g. Chinese for zh
        ds = ds.map(add_language_name_wikilingua)

    dataset_splits = list(ds.keys())
    if subset_name == "xlwic_en_zh":
        # Train set is en; val & test are zh
        dataset_splits.remove("train")

    if split == "validation":
        if split not in dataset_splits or len(dataset_splits) == 1:
            print(f"Validation not found for {ds_name}")
            return
        dataset_splits = ["validation"]
    elif split == "train":
        # Use as much as possible
        # Will need to remove e.g. test datasets to benchmark same task performance
        if len(dataset_splits) > 1 and "validation" in dataset_splits:
            dataset_splits.remove("validation")
        # WikiLingua
        if "sampled_validation" in dataset_splits:
            dataset_splits.remove("sampled_validation")
        if "sampled_test" in dataset_splits:
            dataset_splits.remove("sampled_test")

    if subset_name is None:
        prompt_dataset_name = ds_name
    else:
        subset_name_prompt = subset_name
        if is_wikilingua_cross_lingual:
            # Custom crosslingual prompts
            subset_name_prompt = "en_en"
        elif USE_ENGLISH_PROMPTS and ds_name in DS_TO_ENG_PROMPT:
            subset_name_prompt = DS_TO_ENG_PROMPT[ds_name]
        prompt_dataset_name = f"{ds_name}/{subset_name_prompt}"

    prompts = DatasetTemplates(prompt_dataset_name)

    # TODO: Add capping? (cap = MAX_EXAMPLES_PER_DATASET // num_templates)
    for split in dataset_splits:
        for t_name in prompts.all_template_names:
            if ds_name == "Helsinki-NLP/tatoeba_mt":
                # E.g. translate-this-ara-eng, where eng is the target
                lang_dir = DS_TO_LANG.get(t_name.split("-")[-1].split("_")[0], "en")
            elif ds_name == "allenai/wmt22_african":
                lang_dir = DS_TO_LANG.get(subset_name.split("-")[-1])

            out_path = os.path.join(
                lang_dir, 
                f'xp3_{ds_name}_{subset_name}_{split}_{t_name}.jsonl'.replace("/", "_").replace(" ", "_")
            )
            if os.path.exists(out_path):
                print("Skipping as exists: ", out_path)
                continue
            
            assert len(ds[split]) > 0, f"Got empty: {ds_name}"

            try:
                out_ds = apply_template(dataset=ds[split], template=prompts[t_name])
            except Exception as e:
                print(f"Skipping template due to {e}. DS: {ds_name} Template: {t_name}")
                continue
            # Do not force ascii to allow chars like é
            if len(out_ds) > 0:
                out_ds.to_json(out_path,  orient="records", lines=True, force_ascii=False)


# Testing:
TRAIN_DATASETS = [
    ('super_glue', 'wic'),
    ('pasinit/xlwic', "xlwic_en_zh"),
    ('pasinit/xlwic', "xlwic_fr_fr"),
]
for ds in TRAIN_DATASETS:
    #write_to_jsonl_hub(ds)
    write_to_jsonl_hub(ds, split="train")

#with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#    pool.map(partial(write_to_jsonl_hub, split="train"), TRAIN_DATASETS)
#    pool.map(partial(write_to_jsonl_hub, split="validation"), TRAIN_DATASETS)
