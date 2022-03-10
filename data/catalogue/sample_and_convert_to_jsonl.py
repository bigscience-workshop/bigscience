import argparse
import logging
import re
from pathlib import Path

from datasets import Dataset, load_from_disk
from datasets.utils.logging import set_verbosity_info
from numpy.random import default_rng

set_verbosity_info()
logger = logging.getLogger(__name__)
rng = default_rng(42)

CATALOGUE_DATASETS = {
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_enriched_conllu_ancora_for_ml_training": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_parlament_parla": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_odiencorp": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_book_dash_books": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_uit_vsmec": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_talpco": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_book_dash_books": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vietnamese_students_feedback": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_xquad_ca": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikimedia_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_401_www_elperiodicodemexico_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indonesian_frog_storytelling_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-tum_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indonli": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-bm_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ki_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_viquiquad": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ak_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ts_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-st_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_vilaquad": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_bangla_sentiment_classification_datasets": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ny_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikimedia_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-tw_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_mkb": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-tn_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikimedia_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ln_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-nso_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vinbigdata_asr_vlsp_2020": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wikimedia_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_scielo": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-rn_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vinbigdata_mt_vlsp_2020": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_parallel": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_bppt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-wo_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zht_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-lg_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_502_www_ricemedia_co": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zhs_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-fon_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_recibrew": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-sn_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_arabench": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_30_www_radiocable_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-zu_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_pseudocrawl-filtered_674_ai_baidu_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_ester": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-ig_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_492_www_vivawoman_net": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_bengali_question_answering": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_244_www_df_cl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_scielo": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_485_blog_moneysmart_sg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_488_dailyvanity_sg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_696_www_oercommons_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_483_alvinology_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_kompas": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-xh_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_odiencorp": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_habibi": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_pseudocrawl-filtered_672_pt_globalvoices_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_labr": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_339_www_actasanitaria_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_pseudocrawl-filtered_563_ahotsak_eus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_tempo": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_32_www_elexpresso_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indosum": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_91_www_diario26_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-rw_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_ted_talks_iwslt": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_opus100": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_500_www_asiaone_com_singapore": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_pseudocrawl-filtered_530_www_mediapart_fr": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_470_forums_hardwarezone_com_sg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_223_www_eltambor_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indo4b_jw300": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_219_www_aguasresiduales_info": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_376_www_elpopular_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_62_www_lapagina_com_sv": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_data_on_covid_19_news_coverage_in_vietnam": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_548_remezcla_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_499_www_today_com_news": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vntq_corpus_big": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_pseudocrawl-filtered_545_www_detik_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_pseudocrawl-filtered_637_www_argia_eus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_501_theindependent_sg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_430_www_eldiario_ec": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_420_www_retema_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_kalimat": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_487_thesmartlocal_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_153_financialfood_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikinews_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_158_www_diariodeleon_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_pseudocrawl-filtered_599_fr_globalvoices_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_392_www_muypymes_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_pseudocrawl-filtered_506_goiena_eus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_pib": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_231_ojo_pe": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_167_www_ambientum_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vietnamese_poetry": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_118_www_elheraldo_hn": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_233_www_dinero_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_pseudocrawl-filtered_635_www_berria_eus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_395_www_evwind_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_182_correodelsur_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_56_www_eluniverso_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_project_gutenberg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_250_www_cooperativa_cl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_374_www_talcualdigital_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-yo_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_315_lasillavacia_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_207_elimpulso_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_276_radio_uchile_cl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_325_www_laprensa_hn": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_294_www_laopinion_com_co": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_34_www_losandes_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_90_peru_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_23_www_elconfidencialdigital_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_125_www_noticiasde_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_359_www_efeverde_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_qedcorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_381_www_cuartopoder_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_518_www_elcolombiano_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_280_salamancartvaldia_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_341_es_cointelegraph_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_165_www_ticbeat_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_641_es_globalvoices_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_209_misionesonline_net": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_246_www_eldiarionuevodia_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_86_www_motorpasion_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_257_www_diaridetarragona_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_53_www_expreso_ec": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_367_elcorreoweb_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_159_www_postcrescent_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_422_www_formulatv_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_project_gutenberg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_pseudocrawl-filtered_503_www_zaobao_com_sg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_638_globalvoices_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_78_www_listindiario_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_277_www_entornointeligente_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_404_www_telam_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_431_www_elperiodicoextremadura_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_254_diario_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_169_www_el_carabobeno_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_417_www_radiolaprimerisima_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_21_www_elperiodicodearagon_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_498_www_channelnewsasia_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_304_www_semana_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_386_www_prensalibre_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_urdu-monolingual-corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_286_www_nacion_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_67_www_elpais_cr": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_405_www_emol_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_317_diariocorreo_pe": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-as_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_pseudocrawl-filtered_512_kumparan_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_237_www_cronista_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_116_www_latribuna_hn": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indonesian_news_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_172_www_rionegro_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_226_www_ole_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikiversity_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_287_www_cibercuba_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_pseudocrawl-filtered_572_tirto_id": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_157_www_elsoldemexico_com_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_44_ladiaria_com_uy": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_catalan_government_crawling": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikivoyage_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_213_www_hola_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_324_gestion_pe": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_28_www_fayerwayer_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_nigercongo-sw_aggregated": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_354_www_lagaceta_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_534_www_nairaland_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_253_www_debate_com_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_248_www_telesurtv_net": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_406_www_americaeconomia_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_130_www_elperiodicomediterraneo_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_232_tn_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_leipzig_wortschatz_urdu_newscrawl_2016_sentences": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_71_www_rtve_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_220_www_vanguardia_com_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_pseudocrawl-filtered_549_www_cnnindonesia_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_project_gutenberg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vietai_sat": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_tecla": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_203_www_que_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_320_www_paginasiete_bo": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_181_noticiassin_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_the_pile_europarl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_675_www_elespectador_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_146_www_perfil_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_the_pile_europarl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_the_pile_europarl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_brad_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_79_www_laopiniondemurcia_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_229_www_expansion_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_299_www_lne_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_263_www_lasexta_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_the_pile_europarl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikiquote_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_136_valenciaplaza_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_256_www_laprovincia_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_373_www_farodevigo_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_royal_society_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_245_www_noticiasdenavarra_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_pseudocrawl-filtered_515_www_aajtak_in": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wiktionary_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_58_www_levante_emv_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_198_www_eleconomista_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_249_www_telecinco_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_288_www_marca_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_ksucca": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_189_www_eleconomista_com_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_scielo": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_103_www_elmostrador_cl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ur_leipzig_wortschatz_urdu-pk_web_2019_sentences": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_211_www_elcomercio_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_sanad": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikibooks_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_429_cadenaser_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_267_www_elperiodico_com_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_pseudocrawl-filtered_595_mawdoo3_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_215_www_lainformacion_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh-tw_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh-cn_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_255_elcomercio_pe": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-gu_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-or_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_id_indonesian_news_articles_2017": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_409_www_proceso_com_mx": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_510_timesofindia_indiatimes_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_349_www_eltiempo_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_samanantar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_424_www_lavanguardia_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_100_www_aporrea_org": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_vinbigdata_monolingual_vlsp_2020": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_eu_bsbasque": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_497_www_straitstimes_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_396_www_eldiario_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-pa_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_tashkeela": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-mr_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_du_reader": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_20_www_clarin_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_pseudocrawl-filtered_689_www_abc_net_au": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_pseudocrawl-filtered_667_www_bhaskar_com": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_63_www_lanacion_com_ar": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-kn_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_pseudocrawl-filtered_333_www_elmundo_es": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_project_gutenberg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_pseudocrawl-filtered_550_www_lemonde_fr": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_multi_un_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-te_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_uncorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_multi_un_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_catalan_general_crawling": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_multi_un_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ne_unsupervised_cross_lingual_representation_learning_at_scale": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ml_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_multi_un_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_multi_un_2": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_iitb_english_hindi_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_uncorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_uncorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_no_code_stackexchange": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_uncorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-ta_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_uncorpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_es_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-hi_indic_nlp_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_vi_binhvq_news_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ca_catalan_textual_corpus": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_indic-bn_bangla_lm": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_wikipedia": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_open_subtitles": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_wikisource_filtered": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_openiti_proc": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_ar_arabic_billion_words": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_pt_brwac": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_project_gutenberg": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_the_pile_uspto": 0.5441176470588235, # ((350 [expected] - ( 326 [catalogue_en] - 251 [s2orc] - 21 [uspto]) ) * 1/2 [catalogue_en_proportion]) / (251 [s2orc] + 21 [uspto])
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_code_stackexchange": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_fr_hal_archives_ouvertes": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_code_github-no-gpl": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_zh_wudaocorpora": 1.,
    "/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/lm_en_s2orc_ai2_pdf_parses": 0.5441176470588235, # ((350 [expected] - ( 326 [catalogue_en] - 251 [s2orc] - 21 [uspto]) ) * 1/2 [catalogue_en_proportion]) / (251 [s2orc] + 21 [uspto])
}
OSCAR_DATASETS = {
    # oscar
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/ar": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/bn": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/ca": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/en": 0.13454545454545455, # ((350 [expected] - ( 326 [catalogue_en] - 251 [s2orc] - 21 [uspto]) ) * 1/2 [oscar_en proportion] ) / 1_100 [oscar_en]
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/es": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/eu": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/fr": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/hi": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/id": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/pt": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/ur": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/vi": 1,
    "/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/zh": 1
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", choices=list([Path(path) for path in (set(CATALOGUE_DATASETS.keys()) | set(OSCAR_DATASETS.keys()))]), type=Path, required=True,
        help="Dataset path."
    )
    parser.add_argument(
        "--save-json-dataset-path-prefix", type=Path, required=True,
        help="Where to output json file. Files will be save in `{args.save_jsonl_dataset_path_prefix}/{lang}/{dataset_name}"
    )
    parser.add_argument(
        "--num-proc", type=int, default=1
    )
    parser.add_argument(
        "--batch-size", type=int
    )
    return parser.parse_args()


catalogue_language_regex = re.compile(
    r"^/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/bigscience-catalogue-lm-data/lm_([^_]+)_.*$"
)
normalise_catalogue_dataset_name_regex = re.compile(
    r"^/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data/bigscience-catalogue-lm-data/(.*)$"
)
def get_catalogue_language(dataset_name: str) -> str:
    lang_candidate = catalogue_language_regex.match(dataset_name).group(1)

    # Normalise chinese languages, so that we only consider simplified and traditional chinese as the two chinese languages
    if lang_candidate in ["zh", "zhs", "zh-cn"]:
        lang_candidate = "zhs"
    elif lang_candidate in ["zht", "zh-tw"]:
        lang_candidate = "zht"
    else:
        assert lang_candidate[:2] != "zh"

    return lang_candidate

oscar_to_bs_language = {
    "ar": "ar",
    "bn": "indic-bn",
    "ca": "ca",
    "en": "en",
    "es": "es",
    "eu": "eu",
    "fr": "fr",
    "hi": "indic-hi",
    "id": "id",
    "pt": "pt",
    "ur": "indic-ur",
    "vi": "vi",
    "zh": "zhs"
}
oscar_language_regex = re.compile(
    r"^/gpfsscratch/rech/six/commun/bigscience-datasets/oscar_dedup/(.*)$"
)
def get_oscar_language(dataset_name: str) -> str:
    return oscar_to_bs_language[oscar_language_regex.match(dataset_name).group(1)]


def sample_dataset(dataset: Dataset, ratio: float) -> Dataset:
    logger.info(f"Ratio: {ratio}")
    if ratio >= 1:
        return dataset

    num_samples = int(len(dataset) * ratio)
    indices = rng.choice(len(dataset), size=num_samples, replace=False, shuffle=False)
    return dataset.select(indices)

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    # load_dataset
    logger.info(f"Loading {args.dataset_path}")
    if args.dataset_path in CATALOGUE_DATASETS:
        ds = load_from_disk(args.dataset_path / "final")
    elif args.dataset_path in OSCAR_DATASETS:
        ds = load_from_disk(args.dataset_path)
    else:
        raise NotImplementedError

    # remove all columns except text
    logger.info(f"Removing all columns except `text`")
    columns_to_remove = set(ds.column_names)
    columns_to_remove.remove("text")
    ds = ds.remove_columns(list(columns_to_remove))

    # sample dataset according to ratio
    logger.info(f"Sampling dataset according to given ratio")
    if args.dataset_path in CATALOGUE_DATASETS:
        ds = sample_dataset(ds, CATALOGUE_DATASETS[args.dataset_path])
    elif args.dataset_path in OSCAR_DATASETS:
        ds = sample_dataset(ds, OSCAR_DATASETS[args.dataset_path])
    else:
        raise NotImplementedError

    # save to json
    save_path: Path
    if args.dataset_path in CATALOGUE_DATASETS:
        lang = get_catalogue_language(args.dataset_path)
        filename = f"{normalise_catalogue_dataset_name_regex.match(args.dataset_path).group(1)}.jsonl"
        save_path = args.save_jsonl_dataset_path_prefix / lang / filename
    elif args.dataset_path in OSCAR_DATASETS:
        lang = get_oscar_language(args.dataset_path)
        save_path = args.save_jsonl_dataset_path_prefix / lang / "oscar.jsonl"
    else:
        raise NotImplementedError
    logger.info(f"Saving to {save_path}")
    tmp_save_path = Path(save_path.parent, f"tmp-{save_path.name}")
    tmp_save_path.parent.mkdir(parents=True)
    ds.to_json(
        tmp_save_path,
        num_proc=args.num_proc,
        batch_size=args.batch_size
    )
    tmp_save_path.rename(save_path)

if __name__ == "__main__":
    main()
