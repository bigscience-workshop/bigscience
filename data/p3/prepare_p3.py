from functools import partial
import os
import multiprocessing
from datasets import load_dataset, load_from_disk
import jsonlines

os.environ["HF_DATASETS_CACHE"] = "/gpfsscratch/rech/six/commun/datasets"

"""Get task list:
!git clone https://github.com/bigscience-workshop/t-zero.git
%cd t-zero
!pip install -e .[seqio_tasks]
!pip install -q py7zr

import t0.seqio_tasks
import seqio
tasks = [task.name for task in seqio.MixtureRegistry.get('t0_train').tasks]
print(tasks)
"""
TZERO_TASK_LIST = [
    'adversarial_qa_dbert_answer_the_following_q',
    'adversarial_qa_dbert_based_on',
    'adversarial_qa_dbert_generate_question',
    'adversarial_qa_dbert_question_context_answer',
    'adversarial_qa_dbert_tell_what_it_is',
    'adversarial_qa_dbidaf_answer_the_following_q',
    'adversarial_qa_dbidaf_based_on',
    'adversarial_qa_dbidaf_generate_question',
    'adversarial_qa_dbidaf_question_context_answer',
    'adversarial_qa_dbidaf_tell_what_it_is',
    'adversarial_qa_droberta_answer_the_following_q',
    'adversarial_qa_droberta_based_on',
    'adversarial_qa_droberta_generate_question',
    'adversarial_qa_droberta_question_context_answer',
    'adversarial_qa_droberta_tell_what_it_is',
    'ag_news_classify',
    'ag_news_classify_question_first',
    'ag_news_classify_with_choices',
    'ag_news_classify_with_choices_question_first',
    'ag_news_recommend',
    'ag_news_which_section',
    'ag_news_which_section_choices',
    'amazon_polarity_Is_this_product_review_positive',
    'amazon_polarity_Is_this_review',
    'amazon_polarity_Is_this_review_negative',
    'amazon_polarity_User_recommend_this_product',
    'amazon_polarity_convey_negative_or_positive_sentiment',
    'amazon_polarity_flattering_or_not',
    'amazon_polarity_negative_or_positive_tone',
    'amazon_polarity_user_satisfied',
    'amazon_polarity_would_you_buy',
    'app_reviews_categorize_rating_using_review',
    'app_reviews_convert_to_rating',
    'app_reviews_convert_to_star_rating',
    'app_reviews_generate_review',
    'cnn_dailymail_3.0.0_2_or_3_sentences',
    'cnn_dailymail_3.0.0_generate_story',
    'cnn_dailymail_3.0.0_news_card_view',
    'cnn_dailymail_3.0.0_news_stock',
    'cnn_dailymail_3.0.0_news_summary',
    'cnn_dailymail_3.0.0_spice_up_story',
    'cnn_dailymail_3.0.0_sum_in_brief',
    'cnn_dailymail_3.0.0_tldr_summary',
    'cnn_dailymail_3.0.0_write_an_outline',
    'common_gen_Example_prompt',
    'common_gen_Given_concepts_type_1',
    'common_gen_Given_concepts_type_2',
    'common_gen_Put_together',
    'common_gen_choice_in_concept_centric_sentence_generation',
    'common_gen_random_task_template_prompt',
    'common_gen_sentence_to_concepts',
    'common_gen_topic_to_sentence',
    'common_gen_topics_from_the_sentence',
    'cos_e_v1.11_aligned_with_common_sense',
    'cos_e_v1.11_description_question_option_id',
    'cos_e_v1.11_description_question_option_text',
    'cos_e_v1.11_explain_why_human',
    'cos_e_v1.11_generate_explanation_given_text',
    'cos_e_v1.11_i_think',
    'cos_e_v1.11_question_description_option_id',
    'cos_e_v1.11_question_description_option_text',
    'cos_e_v1.11_question_option_description_id',
    'cos_e_v1.11_question_option_description_text',
    'cos_e_v1.11_rationale',
    'cosmos_qa_context_answer_to_question',
    'cosmos_qa_context_description_question_answer_id',
    'cosmos_qa_context_description_question_answer_text',
    'cosmos_qa_context_description_question_text',
    'cosmos_qa_context_question_description_answer_id',
    'cosmos_qa_context_question_description_answer_text',
    'cosmos_qa_context_question_description_text',
    'cosmos_qa_description_context_question_answer_id',
    'cosmos_qa_description_context_question_answer_text',
    'cosmos_qa_description_context_question_text',
    'cosmos_qa_no_prompt_id',
    'cosmos_qa_no_prompt_text',
    'cosmos_qa_only_question_answer',
    'dbpedia_14_given_a_choice_of_categories_',
    'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to',
    'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to',
    'dbpedia_14_pick_one_category_for_the_following_text',
    'dream_answer_to_dialogue',
    'dream_baseline',
    'dream_generate_first_utterance',
    'dream_generate_last_utterance',
    'dream_read_the_following_conversation_and_answer_the_question',
    'duorc_ParaphraseRC_answer_question',
    'duorc_ParaphraseRC_build_story_around_qa',
    'duorc_ParaphraseRC_decide_worth_it',
    'duorc_ParaphraseRC_extract_answer',
    'duorc_ParaphraseRC_generate_question',
    'duorc_ParaphraseRC_generate_question_by_answer',
    'duorc_ParaphraseRC_movie_director',
    'duorc_ParaphraseRC_question_answering',
    'duorc_ParaphraseRC_title_generation',
    'duorc_SelfRC_answer_question',
    'duorc_SelfRC_build_story_around_qa',
    'duorc_SelfRC_decide_worth_it',
    'duorc_SelfRC_extract_answer',
    'duorc_SelfRC_generate_question',
    'duorc_SelfRC_generate_question_by_answer',
    'duorc_SelfRC_movie_director',
    'duorc_SelfRC_question_answering',
    'duorc_SelfRC_title_generation',
    'gigaword_TLDR',
    'gigaword_first_sentence_title',
    'gigaword_generate_summary_for_this',
    'gigaword_in_a_nutshell',
    'gigaword_make_a_title',
    'gigaword_reverse_writing',
    'gigaword_write_a_title_for_this_sentence',
    'gigaword_write_an_article',
    'gigaword_write_its_sentence',
    'glue_mrpc_equivalent',
    'glue_mrpc_generate_paraphrase',
    'glue_mrpc_generate_sentence',
    'glue_mrpc_paraphrase',
    'glue_mrpc_replace',
    'glue_mrpc_same_thing',
    'glue_mrpc_want_to_know',
    'glue_qqp_answer',
    'glue_qqp_duplicate',
    'glue_qqp_duplicate_or_not',
    'glue_qqp_meaning',
    'glue_qqp_quora',
    'glue_qqp_same_thing',
    'imdb_Movie_Expressed_Sentiment',
    'imdb_Movie_Expressed_Sentiment_2',
    'imdb_Negation_template_for_positive_and_negative',
    'imdb_Reviewer_Enjoyment',
    'imdb_Reviewer_Enjoyment_Yes_No',
    'imdb_Reviewer_Expressed_Sentiment',
    'imdb_Reviewer_Opinion_bad_good_choices',
    'imdb_Reviewer_Sentiment_Feeling',
    'imdb_Sentiment_with_choices_',
    'imdb_Text_Expressed_Sentiment',
    'imdb_Writer_Expressed_Sentiment',
    'kilt_tasks_hotpotqa_combining_facts',
    'kilt_tasks_hotpotqa_complex_question',
    'kilt_tasks_hotpotqa_final_exam',
    'kilt_tasks_hotpotqa_formulate',
    'kilt_tasks_hotpotqa_straighforward_qa',
    'multi_news_distill',
    'multi_news_expand_reverse_task_',
    'multi_news_summarize',
    'multi_news_summary_scenario',
    'multi_news_synthesize',
    'multi_news_what_are_the_key_points',
    'paws_labeled_final_Concatenation',
    'paws_labeled_final_Concatenation_no_label',
    'paws_labeled_final_Meaning',
    'paws_labeled_final_Meaning_no_label',
    'paws_labeled_final_PAWS_ANLI_GPT3',
    'paws_labeled_final_PAWS_ANLI_GPT3_no_label',
    'paws_labeled_final_Rewrite',
    'paws_labeled_final_Rewrite_no_label',
    'paws_labeled_final_context_question',
    'paws_labeled_final_context_question_no_label',
    'paws_labeled_final_paraphrase_task',
    'paws_labeled_final_task_description_no_label',
    'qasc_is_correct_1',
    'qasc_is_correct_2',
    'qasc_qa_with_combined_facts_1',
    'qasc_qa_with_separated_facts_1',
    'qasc_qa_with_separated_facts_2',
    'qasc_qa_with_separated_facts_3',
    'qasc_qa_with_separated_facts_4',
    'qasc_qa_with_separated_facts_5',
    'quail_context_description_question_answer_id',
    'quail_context_description_question_answer_text',
    'quail_context_description_question_text',
    'quail_context_question_answer_description_id',
    'quail_context_question_answer_description_text',
    'quail_context_question_description_answer_id',
    'quail_context_question_description_answer_text',
    'quail_context_question_description_text',
    'quail_description_context_question_answer_id',
    'quail_description_context_question_answer_text',
    'quail_description_context_question_text',
    'quail_no_prompt_id',
    'quail_no_prompt_text',
    'quarel_choose_between',
    'quarel_do_not_use',
    'quarel_heres_a_story',
    'quarel_logic_test',
    'quarel_testing_students',
    'quartz_answer_question_based_on',
    'quartz_answer_question_below',
    'quartz_given_the_fact_answer_the_q',
    'quartz_having_read_above_passage',
    'quartz_paragraph_question_plain_concat',
    'quartz_read_passage_below_choose',
    'quartz_use_info_from_paragraph_question',
    'quartz_use_info_from_question_paragraph',
    'quoref_Answer_Friend_Question',
    'quoref_Answer_Question_Given_Context',
    'quoref_Answer_Test',
    'quoref_Context_Contains_Answer',
    'quoref_Find_Answer',
    'quoref_Found_Context_Online',
    'quoref_Given_Context_Answer_Question',
    'quoref_Guess_Answer',
    'quoref_Guess_Title_For_Context',
    'quoref_Read_And_Extract_',
    'quoref_What_Is_The_Answer',
    'ropes_background_new_situation_answer',
    'ropes_background_situation_middle',
    'ropes_given_background_situation',
    'ropes_new_situation_background_answer',
    'ropes_plain_background_situation',
    'ropes_plain_bottom_hint',
    'ropes_plain_no_background',
    'ropes_prompt_beginning',
    'ropes_prompt_bottom_hint_beginning',
    'ropes_prompt_bottom_no_hint',
    'ropes_prompt_mix',
    'ropes_read_background_situation',
    'rotten_tomatoes_Movie_Expressed_Sentiment',
    'rotten_tomatoes_Movie_Expressed_Sentiment_2',
    'rotten_tomatoes_Reviewer_Enjoyment',
    'rotten_tomatoes_Reviewer_Enjoyment_Yes_No',
    'rotten_tomatoes_Reviewer_Expressed_Sentiment',
    'rotten_tomatoes_Reviewer_Opinion_bad_good_choices',
    'rotten_tomatoes_Reviewer_Sentiment_Feeling',
    'rotten_tomatoes_Sentiment_with_choices_',
    'rotten_tomatoes_Text_Expressed_Sentiment',
    'rotten_tomatoes_Writer_Expressed_Sentiment',
    'samsum_Generate_a_summary_for_this_dialogue',
    'samsum_Given_the_above_dialogue_write_a_summary',
    'samsum_Sum_up_the_following_dialogue',
    'samsum_Summarize_',
    'samsum_Summarize_this_dialogue_',
    'samsum_To_sum_up_this_dialog',
    'samsum_Write_a_dialogue_that_match_this_summary',
    'sciq_Direct_Question',
    'sciq_Direct_Question_Closed_Book_',
    'sciq_Multiple_Choice',
    'sciq_Multiple_Choice_Closed_Book_',
    'sciq_Multiple_Choice_Question_First',
    'social_i_qa_Check_if_a_random_answer_is_valid_or_not',
    'social_i_qa_Generate_answer',
    'social_i_qa_Generate_the_question_from_the_answer',
    'social_i_qa_I_was_wondering',
    'social_i_qa_Show_choices_and_generate_answer',
    'social_i_qa_Show_choices_and_generate_index',
    'trec_fine_grained_ABBR',
    'trec_fine_grained_ABBR_context_first',
    'trec_fine_grained_DESC',
    'trec_fine_grained_DESC_context_first',
    'trec_fine_grained_ENTY',
    'trec_fine_grained_HUM',
    'trec_fine_grained_HUM_context_first',
    'trec_fine_grained_LOC',
    'trec_fine_grained_LOC_context_first',
    'trec_fine_grained_NUM',
    'trec_fine_grained_NUM_context_first',
    'trec_fine_grained_open',
    'trec_fine_grained_open_context_first',
    'trec_pick_the_best_descriptor',
    'trec_trec1',
    'trec_trec2',
    'trec_what_category_best_describe',
    'trec_which_category_best_describes',
    'wiki_bio_comprehension',
    'wiki_bio_guess_person',
    'wiki_bio_key_content',
    'wiki_bio_what_content',
    'wiki_bio_who',
    'wiki_hop_original_choose_best_object_affirmative_1',
    'wiki_hop_original_choose_best_object_affirmative_2',
    'wiki_hop_original_choose_best_object_affirmative_3',
    'wiki_hop_original_choose_best_object_interrogative_1',
    'wiki_hop_original_choose_best_object_interrogative_2',
    'wiki_hop_original_explain_relation',
    'wiki_hop_original_generate_object',
    'wiki_hop_original_generate_subject',
    'wiki_hop_original_generate_subject_and_object',
    'wiki_qa_Decide_good_answer',
    'wiki_qa_Direct_Answer_to_Question',
    'wiki_qa_Generate_Question_from_Topic',
    'wiki_qa_Is_This_True_',
    'wiki_qa_Jeopardy_style',
    'wiki_qa_Topic_Prediction_Answer_Only',
    'wiki_qa_Topic_Prediction_Question_Only',
    'wiki_qa_Topic_Prediction_Question_and_Answer_Pair',
    'wiki_qa_automatic_system',
    'wiki_qa_exercise',
    'wiki_qa_found_on_google',
    'wiqa_does_the_supposed_perturbation_have_an_effect',
    'wiqa_effect_with_label_answer',
    'wiqa_effect_with_string_answer',
    'wiqa_what_is_the_final_step_of_the_following_process',
    'wiqa_what_is_the_missing_first_step',
    'wiqa_what_might_be_the_first_step_of_the_process',
    'wiqa_what_might_be_the_last_step_of_the_process',
    'wiqa_which_of_the_following_is_the_supposed_perturbation',
    'xsum_DOC_boils_down_to_simple_idea_that',
    'xsum_DOC_given_above_write_one_sentence',
    'xsum_DOC_how_would_you_rephrase_few_words',
    'xsum_DOC_tldr',
    'xsum_DOC_write_summary_of_above',
    'xsum_article_DOC_summary',
    'xsum_college_roommate_asked_DOC_so_I_recap',
    'xsum_read_below_DOC_write_abstract',
    'xsum_summarize_DOC',
    'xsum_summarize_this_DOC_summary',
    'yelp_review_full_based_on_that',
    'yelp_review_full_format_rating',
    'yelp_review_full_format_score',
    'yelp_review_full_format_star',
    'yelp_review_full_on_a_scale',
    'yelp_review_full_so_i_would',
    'yelp_review_full_this_place'
]

# Optonally download all first
# for task_name in TZERO_TASK_LIST:
#     ds = load_dataset("bigscience/P3", task_name)

def write_to_jsonl_hub(task_name, split):
    # Could also use ds.to_json()
    ds = load_dataset("bigscience/P3", task_name)
    if split in ds:
        with jsonlines.open(f'p3_{task_name}_{split}.jsonl', mode='w') as writer:
            for example in ds[split].select(range(len(ds[split]))):
                writer.write({
                    "inputs": example["inputs_pretokenized"], 
                    "targets": example["targets_pretokenized"]
                })

def write_to_jsonl_disk(task_name, split):
    ds = load_from_disk(f"{os.environ['six_ALL_CCFRSCRATCH']}/datasets/p3/{task_name}")
    if split in ds:
        with jsonlines.open(f'p3_{task_name}_{split}.jsonl', mode='w') as writer:
            for example in ds[split].select(range(len(ds[split]))):
                writer.write({
                    "inputs": example["inputs_pretokenized"], 
                    "targets": example["targets_pretokenized"]
                })

with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    pool.map(partial(write_to_jsonl_disk, split="train"), TZERO_TASK_LIST)
    pool.map(partial(write_to_jsonl_disk, split="validation"), TZERO_TASK_LIST)


"""
DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/p3t0/p3_t0_validation.jsonl
OUTPUT=/gpfswork/rech/six/commun/bigscience-training/p3t0/p3_t0_validation
TOKENIZER_PATH="bigscience/tokenizer"
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 8

python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 8


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/p31t0/p31t0_train.jsonl
OUTPUT=/gpfswork/rech/six/commun/bigscience-training/p31t0/p31t0_train
TOKENIZER_PATH="bigscience/tokenizer"
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 8
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 8


DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/p31t0/p31t0_validation.jsonl
OUTPUT=/gpfswork/rech/six/commun/bigscience-training/p31t0/p31t0_validation
TOKENIZER_PATH="bigscience/tokenizer"
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 8
python tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --workers 8
"""
