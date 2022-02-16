#!/bin/bash
#SBATCH --job-name=catalogue-jsonl-to-meg-ds # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=10:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/catalogue-jsonl-to-meg-ds/%x-%j.out          # output file name
#SBATCH --account=six@cpu
#SBATCH --array=0-303
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-prod
# We need a specific installation of tokenizers so that it works with bytefallback
conda activate thomas_data_tooling

DATASETS_NAMES=(
bigscience-catalogue-lm-data/lm_ar_arabench
bigscience-catalogue-lm-data/lm_ar_arabic_billion_words
bigscience-catalogue-lm-data/lm_ar_brad_2
bigscience-catalogue-lm-data/lm_ar_habibi
bigscience-catalogue-lm-data/lm_ar_kalimat
bigscience-catalogue-lm-data/lm_ar_ksucca
bigscience-catalogue-lm-data/lm_ar_labr
bigscience-catalogue-lm-data/lm_ar_multi_un_2
bigscience-catalogue-lm-data/lm_ar_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_ar_open_subtitles
bigscience-catalogue-lm-data/lm_ar_openiti
bigscience-catalogue-lm-data/lm_ar_openiti_proc
bigscience-catalogue-lm-data/lm_ar_opus100
bigscience-catalogue-lm-data/lm_ar_osac
bigscience-catalogue-lm-data/lm_ar_qedcorpus
bigscience-catalogue-lm-data/lm_ar_sanad
bigscience-catalogue-lm-data/lm_ar_tashkeela
bigscience-catalogue-lm-data/lm_ar_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_ar_uncorpus
bigscience-catalogue-lm-data/lm_ca_ca_text_corpus
bigscience-catalogue-lm-data/lm_ca_catalan_general_crawling
bigscience-catalogue-lm-data/lm_ca_catalan_government_crawling
bigscience-catalogue-lm-data/lm_ca_catalan_textual_corpus
bigscience-catalogue-lm-data/lm_ca_enriched_conllu_ancora_for_ml_training
bigscience-catalogue-lm-data/lm_ca_open_subtitles
bigscience-catalogue-lm-data/lm_ca_opus100
bigscience-catalogue-lm-data/lm_ca_parlament_parla
bigscience-catalogue-lm-data/lm_ca_sts_ca
bigscience-catalogue-lm-data/lm_ca_teca
bigscience-catalogue-lm-data/lm_ca_tecla
bigscience-catalogue-lm-data/lm_ca_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_ca_tv3_parla
bigscience-catalogue-lm-data/lm_ca_vilaquad
bigscience-catalogue-lm-data/lm_ca_viquiquad
bigscience-catalogue-lm-data/lm_ca_xquad_ca
bigscience-catalogue-lm-data/lm_en_a_million_news_headlines_abc_australia
bigscience-catalogue-lm-data/lm_en_arxiv
bigscience-catalogue-lm-data/lm_en_book_dash_books
bigscience-catalogue-lm-data/lm_en_india_news_headlines_dataset
bigscience-catalogue-lm-data/lm_en_multi_un_2
bigscience-catalogue-lm-data/lm_en_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_en_odiencorp
bigscience-catalogue-lm-data/lm_en_open_subtitles
bigscience-catalogue-lm-data/lm_en_pmc
bigscience-catalogue-lm-data/lm_en_project_gutenberg
bigscience-catalogue-lm-data/lm_en_qedcorpus
bigscience-catalogue-lm-data/lm_en_royal_society_corpus
bigscience-catalogue-lm-data/lm_en_s2orc
bigscience-catalogue-lm-data/lm_en_scielo
bigscience-catalogue-lm-data/lm_en_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_en_the_pile_enron_emails
bigscience-catalogue-lm-data/lm_en_the_pile_europarl
bigscience-catalogue-lm-data/lm_en_the_pile_stack_exchange
bigscience-catalogue-lm-data/lm_en_the_pile_uspto
bigscience-catalogue-lm-data/lm_en_uncorpus
bigscience-catalogue-lm-data/lm_es_multi_un_2
bigscience-catalogue-lm-data/lm_es_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_es_open_subtitles
bigscience-catalogue-lm-data/lm_es_opus100
bigscience-catalogue-lm-data/lm_es_project_gutenberg
bigscience-catalogue-lm-data/lm_es_qedcorpus
bigscience-catalogue-lm-data/lm_es_scielo
bigscience-catalogue-lm-data/lm_es_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_es_the_pile_europarl
bigscience-catalogue-lm-data/lm_es_uncorpus
bigscience-catalogue-lm-data/lm_eu_open_subtitles
bigscience-catalogue-lm-data/lm_eu_opus100
bigscience-catalogue-lm-data/lm_eu_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_fr_bnl_historical_newspapers
bigscience-catalogue-lm-data/lm_fr_book_dash_books
bigscience-catalogue-lm-data/lm_fr_ester
bigscience-catalogue-lm-data/lm_fr_multi_un_2
bigscience-catalogue-lm-data/lm_fr_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_fr_open_subtitles
bigscience-catalogue-lm-data/lm_fr_opus100
bigscience-catalogue-lm-data/lm_fr_project_gutenberg
bigscience-catalogue-lm-data/lm_fr_qedcorpus
bigscience-catalogue-lm-data/lm_fr_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_fr_the_pile_europarl
bigscience-catalogue-lm-data/lm_fr_uncorpus
bigscience-catalogue-lm-data/lm_id_indo4b
bigscience-catalogue-lm-data/lm_id_indonesian_frog_storytelling_corpus
bigscience-catalogue-lm-data/lm_id_indonesian_news_articles_2017
bigscience-catalogue-lm-data/lm_id_indonesian_news_corpus
bigscience-catalogue-lm-data/lm_id_indonli
bigscience-catalogue-lm-data/lm_id_indosum
bigscience-catalogue-lm-data/lm_id_open_subtitles
bigscience-catalogue-lm-data/lm_id_opus100
bigscience-catalogue-lm-data/lm_id_recibrew
bigscience-catalogue-lm-data/lm_id_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_id_twitter_covid19_indonesia_sentiment_analysis
bigscience-catalogue-lm-data/lm_indic-as_opus100
bigscience-catalogue-lm-data/lm_indic-as_samanantar
bigscience-catalogue-lm-data/lm_indic-as_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-bn_bangla_lm
bigscience-catalogue-lm-data/lm_indic-bn_bangla_sentiment_classification_datasets
bigscience-catalogue-lm-data/lm_indic-bn_bengali_question_answering
bigscience-catalogue-lm-data/lm_indic-bn_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-bn_mkb
bigscience-catalogue-lm-data/lm_indic-bn_open_subtitles
bigscience-catalogue-lm-data/lm_indic-bn_opus100
bigscience-catalogue-lm-data/lm_indic-bn_pib
bigscience-catalogue-lm-data/lm_indic-bn_samanantar
bigscience-catalogue-lm-data/lm_indic-bn_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-gu_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-gu_mkb
bigscience-catalogue-lm-data/lm_indic-gu_opus100
bigscience-catalogue-lm-data/lm_indic-gu_pib
bigscience-catalogue-lm-data/lm_indic-gu_samanantar
bigscience-catalogue-lm-data/lm_indic-gu_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-hi_hindi_wikipedia_articles
bigscience-catalogue-lm-data/lm_indic-hi_iitb_english_hindi_corpus
bigscience-catalogue-lm-data/lm_indic-hi_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-hi_lot_of_indic_tweets
bigscience-catalogue-lm-data/lm_indic-hi_mkb
bigscience-catalogue-lm-data/lm_indic-hi_open_subtitles
bigscience-catalogue-lm-data/lm_indic-hi_opus100
bigscience-catalogue-lm-data/lm_indic-hi_pib
bigscience-catalogue-lm-data/lm_indic-hi_qedcorpus
bigscience-catalogue-lm-data/lm_indic-hi_samanantar
bigscience-catalogue-lm-data/lm_indic-hi_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-kn_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-kn_opus100
bigscience-catalogue-lm-data/lm_indic-kn_samanantar
bigscience-catalogue-lm-data/lm_indic-kn_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-ml_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-ml_mkb
bigscience-catalogue-lm-data/lm_indic-ml_open_subtitles
bigscience-catalogue-lm-data/lm_indic-ml_opus100
bigscience-catalogue-lm-data/lm_indic-ml_pib
bigscience-catalogue-lm-data/lm_indic-ml_samanantar
bigscience-catalogue-lm-data/lm_indic-ml_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-mr_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-mr_mkb
bigscience-catalogue-lm-data/lm_indic-mr_opus100
bigscience-catalogue-lm-data/lm_indic-mr_pib
bigscience-catalogue-lm-data/lm_indic-mr_samanantar
bigscience-catalogue-lm-data/lm_indic-mr_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-ne_unsupervised_cross_lingual_representation_learning_at_scale
bigscience-catalogue-lm-data/lm_indic-or_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-or_mkb
bigscience-catalogue-lm-data/lm_indic-or_odiencorp
bigscience-catalogue-lm-data/lm_indic-or_opus100
bigscience-catalogue-lm-data/lm_indic-or_pib
bigscience-catalogue-lm-data/lm_indic-or_samanantar
bigscience-catalogue-lm-data/lm_indic-pa_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-pa_mkb
bigscience-catalogue-lm-data/lm_indic-pa_opus100
bigscience-catalogue-lm-data/lm_indic-pa_pib
bigscience-catalogue-lm-data/lm_indic-pa_samanantar
bigscience-catalogue-lm-data/lm_indic-pa_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-ta_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-ta_mkb
bigscience-catalogue-lm-data/lm_indic-ta_open_subtitles
bigscience-catalogue-lm-data/lm_indic-ta_opus100
bigscience-catalogue-lm-data/lm_indic-ta_pib
bigscience-catalogue-lm-data/lm_indic-ta_samanantar
bigscience-catalogue-lm-data/lm_indic-ta_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-te_indic_nlp_corpus
bigscience-catalogue-lm-data/lm_indic-te_lot_of_indic_tweets
bigscience-catalogue-lm-data/lm_indic-te_mkb
bigscience-catalogue-lm-data/lm_indic-te_open_subtitles
bigscience-catalogue-lm-data/lm_indic-te_opus100
bigscience-catalogue-lm-data/lm_indic-te_pib
bigscience-catalogue-lm-data/lm_indic-te_samanantar
bigscience-catalogue-lm-data/lm_indic-te_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_indic-ur_mkb
bigscience-catalogue-lm-data/lm_indic-ur_open_subtitles
bigscience-catalogue-lm-data/lm_indic-ur_opus100
bigscience-catalogue-lm-data/lm_indic-ur_pib
bigscience-catalogue-lm-data/lm_indic-ur_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_nigercongo-fon_ffr
bigscience-catalogue-lm-data/lm_nigercongo-ig_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-ig_opus100
bigscience-catalogue-lm-data/lm_nigercongo-lg_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-rw_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-rw_opus100
bigscience-catalogue-lm-data/lm_nigercongo-sw_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-sw_swahili_news
bigscience-catalogue-lm-data/lm_nigercongo-sw_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_nigercongo-wo_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-xh_book_dash_books
bigscience-catalogue-lm-data/lm_nigercongo-xh_opus100
bigscience-catalogue-lm-data/lm_nigercongo-yo_masakhaner
bigscience-catalogue-lm-data/lm_nigercongo-yo_opus100
bigscience-catalogue-lm-data/lm_nigercongo-zu_book_dash_books
bigscience-catalogue-lm-data/lm_nigercongo-zu_opus100
bigscience-catalogue-lm-data/lm_pt_brwac
bigscience-catalogue-lm-data/lm_pt_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_pt_open_subtitles
bigscience-catalogue-lm-data/lm_pt_opus100
bigscience-catalogue-lm-data/lm_pt_project_gutenberg
bigscience-catalogue-lm-data/lm_pt_qedcorpus
bigscience-catalogue-lm-data/lm_pt_scielo
bigscience-catalogue-lm-data/lm_pt_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_pt_the_pile_europarl
bigscience-catalogue-lm-data/lm_vi_binhvq_news_corpus
bigscience-catalogue-lm-data/lm_vi_data_on_covid_19_news_coverage_in_vietnam
bigscience-catalogue-lm-data/lm_vi_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_vi_open_subtitles
bigscience-catalogue-lm-data/lm_vi_opus100
bigscience-catalogue-lm-data/lm_vi_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_vi_uit_vsmec
bigscience-catalogue-lm-data/lm_vi_vietai_sat
bigscience-catalogue-lm-data/lm_vi_vietnamese_poetry
bigscience-catalogue-lm-data/lm_vi_vietnamese_students_feedback
bigscience-catalogue-lm-data/lm_vi_vinbigdata_asr_vlsp_2020
bigscience-catalogue-lm-data/lm_vi_vinbigdata_monolingual_vlsp_2020
bigscience-catalogue-lm-data/lm_vi_vinbigdata_mt_vlsp_2020
bigscience-catalogue-lm-data/lm_vi_vntq_corpus_big
bigscience-catalogue-lm-data/lm_zh_du_reader
bigscience-catalogue-lm-data/lm_zh_multi_un_2
bigscience-catalogue-lm-data/lm_zh_open_subtitles
bigscience-catalogue-lm-data/lm_zh_opus100
bigscience-catalogue-lm-data/lm_zh_project_gutenberg
bigscience-catalogue-lm-data/lm_zh_ted_talks_iwslt
bigscience-catalogue-lm-data/lm_zh_uncorpus
bigscience-catalogue-lm-data/lm_zh_wudaocorpora
bigscience-catalogue-lm-data/lm_zh-CN_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_zh-HK_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_zh-TW_multilingual_knowledge_questions_answers
bigscience-catalogue-lm-data/lm_zhs_qedcorpus
bigscience-catalogue-lm-data/lm_zht_qedcorpus
bigscience-catalogue-lm-data/lm_zh_bloom
bigscience-catalogue-lm-data/lm_en_bloom
bigscience-catalogue-lm-data/lm_fr_bloom
bigscience-catalogue-lm-data/lm_id_bloom
bigscience-catalogue-lm-data/lm_pt_bloom
bigscience-catalogue-lm-data/lm_es_bloom
bigscience-catalogue-lm-data/lm_vi_bloom
bigscience-catalogue-lm-data/lm_nigercongo-ak_bloom
bigscience-catalogue-lm-data/lm_nigercongo-bm_bloom
bigscience-catalogue-lm-data/lm_nigercongo-fon_bloom
bigscience-catalogue-lm-data/lm_nigercongo-zu_bloom
bigscience-catalogue-lm-data/lm_nigercongo-rw_bloom
bigscience-catalogue-lm-data/lm_nigercongo-ki_bloom
bigscience-catalogue-lm-data/lm_nigercongo-ln_bloom
bigscience-catalogue-lm-data/lm_nigercongo-lg_bloom
bigscience-catalogue-lm-data/lm_nigercongo-nso_bloom
bigscience-catalogue-lm-data/lm_nigercongo-st_bloom
bigscience-catalogue-lm-data/lm_nigercongo-tn_bloom
bigscience-catalogue-lm-data/lm_nigercongo-sw_bloom
bigscience-catalogue-lm-data/lm_nigercongo-xh_bloom
bigscience-catalogue-lm-data/lm_nigercongo-ts_bloom
bigscience-catalogue-lm-data/lm_indic-bn_bloom
bigscience-catalogue-lm-data/lm_indic-gu_bloom
bigscience-catalogue-lm-data/lm_indic-hi_bloom
bigscience-catalogue-lm-data/lm_indic-kn_bloom
bigscience-catalogue-lm-data/lm_indic-ml_bloom
bigscience-catalogue-lm-data/lm_indic-mr_bloom
bigscience-catalogue-lm-data/lm_indic-or_bloom
bigscience-catalogue-lm-data/lm_indic-pa_bloom
bigscience-catalogue-lm-data/lm_indic-te_bloom
bigscience-catalogue-lm-data/lm_indic-ta_bloom
bigscience-catalogue-lm-data/lm_indic-ur_bloom
bigscience-catalogue-lm-data/lm_eu_bsbasque
bigscience-catalogue-lm-data/lm_indic-ur_urdu-monolingual-corpus
bigscience-catalogue-lm-data/lm_indic-ur_leipzig_wortschatz_urdu_newscrawl_2016_sentences
bigscience-catalogue-lm-data/lm_indic-ur_leipzig_wortschatz_urdu-pk_web_2019_sentences
bigscience-catalogue-lm-data/lm_ar_wikipedia
bigscience-catalogue-lm-data/lm_ca_wikipedia
bigscience-catalogue-lm-data/lm_en_wikipedia
bigscience-catalogue-lm-data/lm_es_wikipedia
bigscience-catalogue-lm-data/lm_eu_wikipedia
bigscience-catalogue-lm-data/lm_fr_wikipedia
bigscience-catalogue-lm-data/lm_id_wikipedia
bigscience-catalogue-lm-data/lm_indic-as_wikipedia
bigscience-catalogue-lm-data/lm_indic-bn_wikipedia
bigscience-catalogue-lm-data/lm_indic-gu_wikipedia
bigscience-catalogue-lm-data/lm_indic-hi_wikipedia
bigscience-catalogue-lm-data/lm_indic-kn_wikipedia
bigscience-catalogue-lm-data/lm_indic-ml_wikipedia
bigscience-catalogue-lm-data/lm_indic-mr_wikipedia
bigscience-catalogue-lm-data/lm_indic-or_wikipedia
bigscience-catalogue-lm-data/lm_indic-pa_wikipedia
bigscience-catalogue-lm-data/lm_indic-ta_wikipedia
bigscience-catalogue-lm-data/lm_indic-te_wikipedia
bigscience-catalogue-lm-data/lm_indic-ur_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ak_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-bm_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ig_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ki_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-lg_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ln_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-nso_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ny_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-rn_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-rw_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-sn_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-st_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-sw_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-tn_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-ts_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-tum_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-tw_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-wo_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-yo_wikipedia
bigscience-catalogue-lm-data/lm_nigercongo-zu_wikipedia
bigscience-catalogue-lm-data/lm_pt_wikipedia
bigscience-catalogue-lm-data/lm_vi_wikipedia
bigscience-catalogue-lm-data/lm_zh-cn_wikipedia
bigscience-catalogue-lm-data/lm_zh-tw_wikipedia
bigscience-catalogue-lm-data/tokenization_nigercongo
bigscience-catalogue-lm-data/lm_code_github
)
DATASET_NAME=${DATASETS_NAMES[$SLURM_ARRAY_TASK_ID]}

TOKENIZER_NAME_OR_PATH=bigscience-catalogue-data-dev/test_fix_empty_string_replacement # TODO: Set it correctly

input=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/jsonl/$DATASET_NAME/data.jsonl
output=$six_ALL_CCFRSCRATCH/bigscience-datasets/catalogue/meg-ds/"$DATASET_NAME"/meg_ds_"${TOKENIZER_NAME_OR_PATH//\//_}"

mkdir -p $(dirname $output)

if [[ -f "$output"_text_document.bin ]];
then
    echo "$output exists."
    exit 0
fi

echo "Downloading ${DATASET_NAME}"
export HF_DATASETS_OFFLINE=1

cd $six_ALL_CCFRWORK/code/Megatron-DeepSpeed
/usr/bin/time -v python tools/preprocess_data.py \
       --input $input \
       --output-prefix $output \
       --dataset-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
       --append-eod \
       --workers 40
