#!/bin/bash
#SBATCH --job-name=evaluate_t0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --time 5:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-158

# XNLITR:
# --array=0-79

# Fixed prompts subset:
# --array=0-63

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate thomas_t_zero_evaluation

#CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-prefix
CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3/bloom-6b3
#CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31
#CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmixnewcodelonglossseq
#CHECKPOINT_PATH=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/bloom7b1zz

WORKDIR=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0

pushd $WORKDIR

OUTPUT_DIR=$CHECKPOINT_PATH/evaluation
mkdir -p $OUTPUT_DIR


# Validation
DATASETS_AND_CONFIGS_VAL=(
head_qa,en,en,"multiple_choice_q_and_a_index_with_context_en"
head_qa,en,en,"multiple_choice_q_and_a_en"
head_qa,en,en,"multiple_choice_q_and_a_index_en"
head_qa,en,en,"multiple_choice_a_and_q_with_context_en"
head_qa,en,en,"multiple_choice_a_and_q_en"
head_qa,es,en,"multiple_choice_q_and_a_index_with_context_en"
head_qa,es,en,"multiple_choice_q_and_a_en"
head_qa,es,en,"multiple_choice_q_and_a_index_en"
head_qa,es,en,"multiple_choice_a_and_q_with_context_en"
head_qa,es,en,"multiple_choice_a_and_q_en"
climate_fever,None,None,"first_evidence_and_claim_itemization"
climate_fever,None,None,"claim_and_all_supporting_evidences"
climate_fever,None,None,"fifth_evidence_and_claim_itemization"
climate_fever,None,None,"third_evidence_claim_pair"
climate_fever,None,None,"second_evidence_and_claim_itemization"
codah,codah,None,"interrogative_instruction_after_sentence_and_choices"
codah,codah,None,"affirmative_instruction_before_sentence_and_choices"
codah,codah,None,"affirmative_instruction_after_sentence_and_choices"
aqua_rat,raw,None,"select_the_best_option"
aqua_rat,raw,None,"answer_quiz"
aqua_rat,raw,None,"Answer questions from options"
commonsense_qa,None,None,"answer_given_question_without_options"
commonsense_qa,None,None,"question_answering"
commonsense_qa,None,None,"most_suitable_answer"
)

DATASETS_AND_CONFIGS_VAL=(
amazon_reviews_multi,en,en,"prompt_title_to_star",validation
amazon_reviews_multi,en,en,"prompt_review_to_star",validation
amazon_reviews_multi,en,en,"prompt_body_title_to_star",validation
amazon_reviews_multi,en,zh,"prompt_title_to_star",validation
amazon_reviews_multi,en,zh,"prompt_review_to_star",validation
amazon_reviews_multi,en,zh,"prompt_body_title_to_star",validation
amazon_reviews_multi,en,fr,"prompt_title_to_star",validation
amazon_reviews_multi,en,fr,"prompt_review_to_star",validation
amazon_reviews_multi,en,fr,"prompt_body_title_to_star",validation
amazon_reviews_multi,en,es,"prompt_title_to_star",validation
amazon_reviews_multi,en,es,"prompt_review_to_star",validation
amazon_reviews_multi,en,es,"prompt_body_title_to_star",validation
amazon_us_reviews,Wireless_v1_00,None,"Given the review headline return a categorical rating",validation
amazon_us_reviews,Wireless_v1_00,None,"Given the review body return a categorical rating",validation
art,None,None,"choose_hypothesis_options",validation
art,None,None,"choose_hypothesis_believable",validation
art,None,None,"choose_hypothesis",validation
art,None,None,"choose_hypothesis_desc",validation
art,None,None,"choose_hypothesis_likely",validation
banking77,None,None,"help_page_topic",test
banking77,None,None,"direct_to_which_department",test
banking77,None,None,"choose the correct department",test
banking77,None,None,"rephrase_as_banking_term",test
banking77,None,None,"choose_the_correct_intent",test
blbooksgenre,title_genre_classifiction,None,"multi-choice",train
blbooksgenre,title_genre_classifiction,None,"premise_context_first",train
blbooksgenre,title_genre_classifiction,None,"classify",train
blimp,adjunct_island,None,"grammatical_between_1_2",train
blimp,adjunct_island,None,"grammatical_between_A_B",train
blimp,adjunct_island,None,"grammatical_which_one_1_2",train
blimp,adjunct_island,None,"single_sentence_bad_yes_no",train
blimp,adjunct_island,None,"single_sentence_good_yes_no",train
cbt,CN,None,"Fill Blank with Options - Replace",validation
cbt,CN,None,"Fill Blank with Options - In the following",validation
cbt,CN,None,"Fill Blank without Options",validation
cbt,CN,None,"Fill Blanks with Options - above story",validation
cbt,CN,None,"Fill Blank with Options - Which",validation
conv_ai,None,None,"engagement_alice_interested",train
conv_ai,None,None,"engagement_bob_real",train
conv_ai,None,None,"engagement_question_before",train
conv_ai,None,None,"engagement_bob_interested",train
conv_ai,None,None,"engagement_question_after",train
conv_ai_2,None,None,"match_profile_question",train
conv_ai_2,None,None,"match_profile_expected",train
conv_ai_2,None,None,"match_profile_guess",train
conv_ai_2,None,None,"match_profile_agree",train
conv_ai_2,None,None,"match_profile",train
conv_ai_3,None,None,"clarification_needed",validation
conv_ai_3,None,None,"score_give_number",validation
conv_ai_3,None,None,"ambiguous",validation
conv_ai_3,None,None,"directly_answer",validation
conv_ai_3,None,None,"score_how_much",validation
craigslist_bargains,None,None,"good deal for seller no list price implicit",validation
craigslist_bargains,None,None,"good deal for seller no list price",validation
craigslist_bargains,None,None,"good deal for seller",validation
craigslist_bargains,None,None,"best deal",validation
ecthr_cases,alleged-violation-prediction,None,"implicit_advice_number",validation
ecthr_cases,alleged-violation-prediction,None,"ecthr_alleged_articles_declaration_at_end",validation
ecthr_cases,alleged-violation-prediction,None,"ecthr_alleged_articles_question_at_start",validation
ecthr_cases,alleged-violation-prediction,None,"implicit_judgment_paragraph",validation
ecthr_cases,alleged-violation-prediction,None,"confirm number of violated articles",validation
emo,None,None,"persons_describe",validation
emo,None,None,"final_message",validation
emo,None,None,"what_emotion_do_you_think",validation
emo,None,None,"emotional_state",validation
emo,None,None,"dialogue_between",validation
emotion,None,None,"choose_the_best_emotion_label",test
emotion,None,None,"reply_with_emoation_label",test
emotion,None,None,"answer_with_class_label",test
emotion,None,None,"answer_question_with_emotion_label",test
emotion,None,None,"select_emotion_label_from_list",test
financial_phrasebank,sentences_allagree,None,"share_price_option",train
financial_phrasebank,sentences_allagree,None,"sentiment",train
financial_phrasebank,sentences_allagree,None,"word_comes_to_mind",train
financial_phrasebank,sentences_allagree,None,"complementary_industries",train
financial_phrasebank,sentences_allagree,None,"bullish_neutral_bearish",train
glue,cola,None,"Make sense yes no",validation
glue,cola,None,"is_this_correct",validation
glue,cola,None,"editing",validation
glue,cola,None,"Following sentence acceptable",validation
glue,cola,None,"Previous sentence acceptable",validation
glue,sst2,None,"positive negative after",validation
glue,sst2,None,"review",validation
glue,sst2,None,"said",validation
glue,sst2,None,"following positive negative",validation
glue,sst2,None,"happy or mad",validation
health_fact,None,None,"claim_veracity_classification_after_reading_I_believe",validation
health_fact,None,None,"claim_explanation_classification",validation
health_fact,None,None,"claim_veracity_classification_tell_me",validation
hlgd,None,None,"is_same_event_with_time_interrogative_related",validation
hlgd,None,None,"is_same_event_interrogative_talk",validation
hlgd,None,None,"is_same_event_with_time_interrogative_talk",validation
hlgd,None,None,"is_same_event_refer",validation
hlgd,None,None,"is_same_event_editor_asks",validation
hyperpartisan_news_detection,byarticle,None,"consider_does_it_follow_a_hyperpartisan_argumentation",train
hyperpartisan_news_detection,byarticle,None,"follows_hyperpartisan_argumentation",train
hyperpartisan_news_detection,byarticle,None,"consume_with_caution",train
hyperpartisan_news_detection,byarticle,None,"extreme_left_wing_or_right_wing",train
hyperpartisan_news_detection,byarticle,None,"consider_it_exhibits_extreme_one_sidedness",train
liar,None,None,"Guess affiliation",validation
liar,None,None,"Given statement guess category",validation
liar,None,None,"Given statement guess subject ",validation
lince,sa_spaeng,None,"original poster expressed sentiment",validation
lince,sa_spaeng,None,"sentiment trying to express",validation
lince,sa_spaeng,None,"express sentiment",validation
lince,sa_spaeng,None,"negation template",validation
lince,sa_spaeng,None,"the author seem",validation
math_qa,None,None,"choose_correct_og",test
math_qa,None,None,"pick_the_correct",test
math_qa,None,None,"first_choice_then_problem",test
math_qa,None,None,"problem_set_type",test
math_qa,None,None,"gre_problem",test
movie_rationales,None,None,"Standard binary sentiment analysis",validation
movie_rationales,None,None,"Evidences sentiment classification",validation
movie_rationales,None,None,"Evidences + review",validation
movie_rationales,None,None,"Generate evidences and sentiment",validation
mwsc,None,None,"in-the-sentence-question-first",validation
mwsc,None,None,"what-think",validation
mwsc,None,None,"in-the-sentence",validation
mwsc,None,None,"options-or",validation
mwsc,None,None,"is-correct",validation
poem_sentiment,None,None,"positive_or_negative_sentiment_variation_2",validation
poem_sentiment,None,None,"question_answer_format",validation
poem_sentiment,None,None,"guess_sentiment_without_options_variation_1",validation
poem_sentiment,None,None,"positive_or_negative_sentiment_variation_1",validation
poem_sentiment,None,None,"most_appropriate_sentiment",validation
onestop_english,None,None,"esl_context",train
onestop_english,None,None,"ara_context",train
onestop_english,None,None,"determine_reading_level_from_the_first_three_sentences",train
onestop_english,None,None,"esl_variation",train
onestop_english,None,None,"assess",train
pubmed_qa,pqa_labeled,None,"Long Answer to Final Decision",train
pubmed_qa,pqa_labeled,None,"Context Section Type",train
pubmed_qa,pqa_labeled,None,"Question Answering (Short)",train
riddle_sense,None,None,"answer_to_question",validation
riddle_sense,None,None,"most_suitable_answer",validation
riddle_sense,None,None,"answer_given_question_without_options",validation
riddle_sense,None,None,"question_to_answer_index",validation
riddle_sense,None,None,"question_answering",validation
scicite,None,None,"Classify intent w/section (select choice)",validation
scicite,None,None,"Classify intent (choices first)",validation
scicite,None,None,"Classify intent (select choice)",validation
scicite,None,None,"Classify intent",validation
scicite,None,None,"can_describe",validation
selqa,answer_selection_analysis,None,"is-he-talking-about",validation
selqa,answer_selection_analysis,None,"would-make-sense-qu-rand",validation
selqa,answer_selection_analysis,None,"about-topic-vs-random",validation
selqa,answer_selection_analysis,None,"make-sense-rand",validation
selqa,answer_selection_analysis,None,"which-answer-1st-vs-random",validation
snips_built_in_intents,None,None,"voice_intent",train
snips_built_in_intents,None,None,"categorize_query",train
snips_built_in_intents,None,None,"intent_query",train
snips_built_in_intents,None,None,"categorize_query_brief",train
snips_built_in_intents,None,None,"query_intent",train
turk,None,None,"simplify",test
turk,None,None,"choose-verbose",test
turk,None,None,"choose-simplification",test
)


# Fixed prompts subset:
DATASETS_AND_CONFIGS_L1=(
super_glue,copa,None,"best_option"
super_glue,copa,None,"i_am_hesitating"
super_glue,copa,None,"cause_effect"
super_glue,copa,None,"plausible_alternatives"
xcopa,id,en,"best_option"
xcopa,id,en,"i_am_hesitating"
xcopa,id,en,"cause_effect"
xcopa,id,en,"plausible_alternatives"
xcopa,sw,en,"best_option"
xcopa,sw,en,"i_am_hesitating"
xcopa,sw,en,"cause_effect"
xcopa,sw,en,"plausible_alternatives"
xcopa,ta,en,"best_option"
xcopa,ta,en,"i_am_hesitating"
xcopa,ta,en,"cause_effect"
xcopa,ta,en,"plausible_alternatives"
xcopa,vi,en,"best_option"
xcopa,vi,en,"i_am_hesitating"
xcopa,vi,en,"cause_effect"
xcopa,vi,en,"plausible_alternatives"
xcopa,zh,en,"best_option"
xcopa,zh,en,"i_am_hesitating"
xcopa,zh,en,"cause_effect"
xcopa,zh,en,"plausible_alternatives"
story_cloze,2016,None,"Story Continuation and Options"
story_cloze,2016,None,"Answer Given options"
story_cloze,2016,None,"Novel Correct Ending"
story_cloze,2016,None,"Choose Story Ending"
Muennighoff/xstory_cloze,ar,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,ar,en,"Answer Given options"
Muennighoff/xstory_cloze,ar,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,ar,en,"Choose Story Ending"
Muennighoff/xstory_cloze,es,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,es,en,"Answer Given options"
Muennighoff/xstory_cloze,es,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,es,en,"Choose Story Ending"
Muennighoff/xstory_cloze,eu,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,eu,en,"Answer Given options"
Muennighoff/xstory_cloze,eu,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,eu,en,"Choose Story Ending"
Muennighoff/xstory_cloze,id,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,id,en,"Answer Given options"
Muennighoff/xstory_cloze,id,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,id,en,"Choose Story Ending"
Muennighoff/xstory_cloze,hi,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,hi,en,"Answer Given options"
Muennighoff/xstory_cloze,hi,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,hi,en,"Choose Story Ending"
Muennighoff/xstory_cloze,zh,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,zh,en,"Answer Given options"
Muennighoff/xstory_cloze,zh,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,zh,en,"Choose Story Ending"
winogrande,winogrande_xl,None,"Replace"
winogrande,winogrande_xl,None,"stand for"
winogrande,winogrande_xl,None,"True or False"
Muennighoff/xwinograd,fr,en,"Replace"
Muennighoff/xwinograd,fr,en,"stand for"
Muennighoff/xwinograd,fr,en,"True or False"
Muennighoff/xwinograd,pt,en,"Replace"
Muennighoff/xwinograd,pt,en,"stand for"
Muennighoff/xwinograd,pt,en,"True or False"
Muennighoff/xwinograd,zh,en,"Replace"
Muennighoff/xwinograd,zh,en,"stand for"
Muennighoff/xwinograd,zh,en,"True or False"
)

# All:
DATASETS_AND_CONFIGS_L1=(
super_glue,copa,None,"best_option"
super_glue,copa,None,"C1 or C2? premise, so/because…"
super_glue,copa,None,"i_am_hesitating"
super_glue,copa,None,"cause_effect"
super_glue,copa,None,"plausible_alternatives"
super_glue,rte,None,"MNLI crowdsource"
super_glue,rte,None,"GPT-3 style"
super_glue,rte,None,"does it follow that"
super_glue,rte,None,"should assume"
super_glue,rte,None,"guaranteed true"
anli,dev_r1,None,"guaranteed/possible/impossible"
anli,dev_r1,None,"MNLI crowdsource"
anli,dev_r1,None,"GPT-3 style"
anli,dev_r1,None,"justified in saying"
anli,dev_r1,None,"can we infer"
anli,dev_r2,None,"guaranteed/possible/impossible"
anli,dev_r2,None,"MNLI crowdsource"
anli,dev_r2,None,"GPT-3 style"
anli,dev_r2,None,"justified in saying"
anli,dev_r2,None,"can we infer"
anli,dev_r3,None,"guaranteed/possible/impossible"
anli,dev_r3,None,"MNLI crowdsource"
anli,dev_r3,None,"GPT-3 style"
anli,dev_r3,None,"justified in saying"
anli,dev_r3,None,"can we infer"
super_glue,cb,None,"guaranteed/possible/impossible"
super_glue,cb,None,"MNLI crowdsource"
super_glue,cb,None,"GPT-3 style"
super_glue,cb,None,"justified in saying"
super_glue,cb,None,"can we infer"
super_glue,rte,None,"MNLI crowdsource"
super_glue,rte,None,"GPT-3 style"
super_glue,rte,None,"does it follow that"
super_glue,rte,None,"should assume"
super_glue,rte,None,"guaranteed true"
winogrande,winogrande_xl,None,"underscore refer to"
winogrande,winogrande_xl,None,"Replace"
winogrande,winogrande_xl,None,"stand for"
winogrande,winogrande_xl,None,"does underscore refer to"
winogrande,winogrande_xl,None,"True or False"
story_cloze,2016,None,"Story Continuation and Options"
story_cloze,2016,None,"Answer Given options"
story_cloze,2016,None,"Novel Correct Ending"
story_cloze,2016,None,"Generate Ending"
story_cloze,2016,None,"Choose Story Ending"
Muennighoff/xstory_cloze,ar,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,ar,en,"Answer Given options"
Muennighoff/xstory_cloze,ar,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,ar,en,"Generate Ending"
Muennighoff/xstory_cloze,ar,en,"Choose Story Ending"
Muennighoff/xstory_cloze,es,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,es,en,"Answer Given options"
Muennighoff/xstory_cloze,es,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,es,en,"Generate Ending"
Muennighoff/xstory_cloze,es,en,"Choose Story Ending"
Muennighoff/xstory_cloze,eu,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,eu,en,"Answer Given options"
Muennighoff/xstory_cloze,eu,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,eu,en,"Generate Ending"
Muennighoff/xstory_cloze,eu,en,"Choose Story Ending"
Muennighoff/xstory_cloze,id,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,id,en,"Answer Given options"
Muennighoff/xstory_cloze,id,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,id,en,"Generate Ending"
Muennighoff/xstory_cloze,id,en,"Choose Story Ending"
Muennighoff/xstory_cloze,hi,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,hi,en,"Answer Given options"
Muennighoff/xstory_cloze,hi,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,hi,en,"Generate Ending"
Muennighoff/xstory_cloze,hi,en,"Choose Story Ending"
Muennighoff/xstory_cloze,zh,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,zh,en,"Answer Given options"
Muennighoff/xstory_cloze,zh,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,zh,en,"Generate Ending"
Muennighoff/xstory_cloze,zh,en,"Choose Story Ending"
xnli,ar,en,"guaranteed/possible/impossible"
xnli,ar,en,"MNLI crowdsource"
xnli,ar,en,"GPT-3 style"
xnli,ar,en,"justified in saying"
xnli,ar,en,"can we infer"
xnli,en,en,"guaranteed/possible/impossible"
xnli,en,en,"MNLI crowdsource"
xnli,en,en,"GPT-3 style"
xnli,en,en,"justified in saying"
xnli,en,en,"can we infer"
xnli,es,en,"guaranteed/possible/impossible"
xnli,es,en,"MNLI crowdsource"
xnli,es,en,"GPT-3 style"
xnli,es,en,"justified in saying"
xnli,es,en,"can we infer"
xnli,fr,en,"guaranteed/possible/impossible"
xnli,fr,en,"MNLI crowdsource"
xnli,fr,en,"GPT-3 style"
xnli,fr,en,"justified in saying"
xnli,fr,en,"can we infer"
xnli,hi,en,"guaranteed/possible/impossible"
xnli,hi,en,"MNLI crowdsource"
xnli,hi,en,"GPT-3 style"
xnli,hi,en,"justified in saying"
xnli,hi,en,"can we infer"
xnli,sw,en,"guaranteed/possible/impossible"
xnli,sw,en,"MNLI crowdsource"
xnli,sw,en,"GPT-3 style"
xnli,sw,en,"justified in saying"
xnli,sw,en,"can we infer"
xnli,ur,en,"guaranteed/possible/impossible"
xnli,ur,en,"MNLI crowdsource"
xnli,ur,en,"GPT-3 style"
xnli,ur,en,"justified in saying"
xnli,ur,en,"can we infer"
xnli,vi,en,"guaranteed/possible/impossible"
xnli,vi,en,"MNLI crowdsource"
xnli,vi,en,"GPT-3 style"
xnli,vi,en,"justified in saying"
xnli,vi,en,"can we infer"
xnli,zh,en,"guaranteed/possible/impossible"
xnli,zh,en,"MNLI crowdsource"
xnli,zh,en,"GPT-3 style"
xnli,zh,en,"justified in saying"
xnli,zh,en,"can we infer"
xcopa,id,en,"best_option"
xcopa,id,en,"C1 or C2? premise, so/because…"
xcopa,id,en,"i_am_hesitating"
xcopa,id,en,"cause_effect"
xcopa,id,en,"plausible_alternatives"
xcopa,sw,en,"best_option"
xcopa,sw,en,"C1 or C2? premise, so/because…"
xcopa,sw,en,"i_am_hesitating"
xcopa,sw,en,"cause_effect"
xcopa,sw,en,"plausible_alternatives"
xcopa,ta,en,"best_option"
xcopa,ta,en,"C1 or C2? premise, so/because…"
xcopa,ta,en,"i_am_hesitating"
xcopa,ta,en,"cause_effect"
xcopa,ta,en,"plausible_alternatives"
xcopa,vi,en,"best_option"
xcopa,vi,en,"C1 or C2? premise, so/because…"
xcopa,vi,en,"i_am_hesitating"
xcopa,vi,en,"cause_effect"
xcopa,vi,en,"plausible_alternatives"
xcopa,zh,en,"best_option"
xcopa,zh,en,"C1 or C2? premise, so/because…"
xcopa,zh,en,"i_am_hesitating"
xcopa,zh,en,"cause_effect"
xcopa,zh,en,"plausible_alternatives"
Muennighoff/xwinograd,en,en,"underscore refer to"
Muennighoff/xwinograd,en,en,"Replace"
Muennighoff/xwinograd,en,en,"stand for"
Muennighoff/xwinograd,en,en,"does underscore refer to"
Muennighoff/xwinograd,en,en,"True or False"
Muennighoff/xwinograd,fr,en,"underscore refer to"
Muennighoff/xwinograd,fr,en,"Replace"
Muennighoff/xwinograd,fr,en,"stand for"
Muennighoff/xwinograd,fr,en,"does underscore refer to"
Muennighoff/xwinograd,fr,en,"True or False"
Muennighoff/xwinograd,pt,en,"underscore refer to"
Muennighoff/xwinograd,pt,en,"Replace"
Muennighoff/xwinograd,pt,en,"stand for"
Muennighoff/xwinograd,pt,en,"does underscore refer to"
Muennighoff/xwinograd,pt,en,"True or False"
Muennighoff/xwinograd,zh,en,"underscore refer to"
Muennighoff/xwinograd,zh,en,"Replace"
Muennighoff/xwinograd,zh,en,"stand for"
Muennighoff/xwinograd,zh,en,"does underscore refer to"
Muennighoff/xwinograd,zh,en,"True or False"
)

DATASETS_AND_CONFIGS_L2=(
Muennighoff/xstory_cloze,ru,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,ru,en,"Answer Given options"
Muennighoff/xstory_cloze,ru,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,ru,en,"Generate Ending"
Muennighoff/xstory_cloze,ru,en,"Choose Story Ending"
Muennighoff/xstory_cloze,my,en,"Story Continuation and Options"
Muennighoff/xstory_cloze,my,en,"Answer Given options"
Muennighoff/xstory_cloze,my,en,"Novel Correct Ending"
Muennighoff/xstory_cloze,my,en,"Generate Ending"
Muennighoff/xstory_cloze,my,en,"Choose Story Ending"
xnli,bg,en,"guaranteed/possible/impossible"
xnli,bg,en,"MNLI crowdsource"
xnli,bg,en,"GPT-3 style"
xnli,bg,en,"justified in saying"
xnli,bg,en,"can we infer"
xnli,de,en,"guaranteed/possible/impossible"
xnli,de,en,"MNLI crowdsource"
xnli,de,en,"GPT-3 style"
xnli,de,en,"justified in saying"
xnli,de,en,"can we infer"
xnli,el,en,"guaranteed/possible/impossible"
xnli,el,en,"MNLI crowdsource"
xnli,el,en,"GPT-3 style"
xnli,el,en,"justified in saying"
xnli,el,en,"can we infer"
xnli,ru,en,"guaranteed/possible/impossible"
xnli,ru,en,"MNLI crowdsource"
xnli,ru,en,"GPT-3 style"
xnli,ru,en,"justified in saying"
xnli,ru,en,"can we infer"
xnli,th,en,"guaranteed/possible/impossible"
xnli,th,en,"MNLI crowdsource"
xnli,th,en,"GPT-3 style"
xnli,th,en,"justified in saying"
xnli,th,en,"can we infer"
xnli,tr,en,"guaranteed/possible/impossible"
xnli,tr,en,"MNLI crowdsource"
xnli,tr,en,"GPT-3 style"
xnli,tr,en,"justified in saying"
xnli,tr,en,"can we infer"
Muennighoff/xwinograd,ru,en,"underscore refer to"
Muennighoff/xwinograd,ru,en,"Replace"
Muennighoff/xwinograd,ru,en,"stand for"
Muennighoff/xwinograd,ru,en,"does underscore refer to"
Muennighoff/xwinograd,ru,en,"True or False"
Muennighoff/xwinograd,jp,en,"underscore refer to"
Muennighoff/xwinograd,jp,en,"Replace"
Muennighoff/xwinograd,jp,en,"stand for"
Muennighoff/xwinograd,jp,en,"does underscore refer to"
Muennighoff/xwinograd,jp,en,"True or False"
xcopa,et,en,"best_option"
xcopa,et,en,"C1 or C2? premise, so/because…"
xcopa,et,en,"i_am_hesitating"
xcopa,et,en,"cause_effect"
xcopa,et,en,"plausible_alternatives"
xcopa,ht,en,"best_option"
xcopa,ht,en,"C1 or C2? premise, so/because…"
xcopa,ht,en,"i_am_hesitating"
xcopa,ht,en,"cause_effect"
xcopa,ht,en,"plausible_alternatives"
xcopa,it,en,"best_option"
xcopa,it,en,"C1 or C2? premise, so/because…"
xcopa,it,en,"i_am_hesitating"
xcopa,it,en,"cause_effect"
xcopa,it,en,"plausible_alternatives"
xcopa,qu,en,"best_option"
xcopa,qu,en,"C1 or C2? premise, so/because…"
xcopa,qu,en,"i_am_hesitating"
xcopa,qu,en,"cause_effect"
xcopa,qu,en,"plausible_alternatives"
xcopa,th,en,"best_option"
xcopa,th,en,"C1 or C2? premise, so/because…"
xcopa,th,en,"i_am_hesitating"
xcopa,th,en,"cause_effect"
xcopa,th,en,"plausible_alternatives"
xcopa,tr,en,"best_option"
xcopa,tr,en,"C1 or C2? premise, so/because…"
xcopa,tr,en,"i_am_hesitating"
xcopa,tr,en,"cause_effect"
xcopa,tr,en,"plausible_alternatives"
)

DATASETS_AND_CONFIGS_XNLITR=(
xnli,ar,ar,"guaranteed/possible/impossible_arht"
xnli,ar,ar,"MNLI crowdsource_arht"
xnli,ar,ar,"GPT-3 style_arht"
xnli,ar,ar,"justified in saying_arht"
xnli,ar,ar,"can we infer_arht"
xnli,ar,ar,"guaranteed/possible/impossible_armt"
xnli,ar,ar,"MNLI crowdsource_armt"
xnli,ar,ar,"GPT-3 style_armt"
xnli,ar,ar,"justified in saying_armt"
xnli,ar,ar,"can we infer_armt"
xnli,es,es,"guaranteed/possible/impossible_esht"
xnli,es,es,"MNLI crowdsource_esht"
xnli,es,es,"GPT-3 style_esht"
xnli,es,es,"justified in saying_esht"
xnli,es,es,"can we infer_esht"
xnli,es,es,"guaranteed/possible/impossible_esmt"
xnli,es,es,"MNLI crowdsource_esmt"
xnli,es,es,"GPT-3 style_esmt"
xnli,es,es,"justified in saying_esmt"
xnli,es,es,"can we infer_esmt"
xnli,fr,fr,"guaranteed/possible/impossible_frht"
xnli,fr,fr,"MNLI crowdsource_frht"
xnli,fr,fr,"GPT-3 style_frht"
xnli,fr,fr,"justified in saying_frht"
xnli,fr,fr,"can we infer_frht"
xnli,fr,fr,"guaranteed/possible/impossible_frmt"
xnli,fr,fr,"MNLI crowdsource_frmt"
xnli,fr,fr,"GPT-3 style_frmt"
xnli,fr,fr,"justified in saying_frmt"
xnli,fr,fr,"can we infer_frmt"
xnli,hi,hi,"guaranteed/possible/impossible_hiht"
xnli,hi,hi,"MNLI crowdsource_hiht"
xnli,hi,hi,"GPT-3 style_hiht"
xnli,hi,hi,"justified in saying_hiht"
xnli,hi,hi,"can we infer_hiht"
xnli,hi,hi,"guaranteed/possible/impossible_himt"
xnli,hi,hi,"MNLI crowdsource_himt"
xnli,hi,hi,"GPT-3 style_himt"
xnli,hi,hi,"justified in saying_himt"
xnli,hi,hi,"can we infer_himt"
xnli,ur,ur,"guaranteed/possible/impossible_urht"
xnli,ur,ur,"MNLI crowdsource_urht"
xnli,ur,ur,"GPT-3 style_urht"
xnli,ur,ur,"justified in saying_urht"
xnli,ur,ur,"can we infer_urht"
xnli,ur,ur,"guaranteed/possible/impossible_urmt"
xnli,ur,ur,"MNLI crowdsource_urmt"
xnli,ur,ur,"GPT-3 style_urmt"
xnli,ur,ur,"justified in saying_urmt"
xnli,ur,ur,"can we infer_urmt"
xnli,sw,sw,"guaranteed/possible/impossible_swht"
xnli,sw,sw,"MNLI crowdsource_swht"
xnli,sw,sw,"GPT-3 style_swht"
xnli,sw,sw,"justified in saying_swht"
xnli,sw,sw,"can we infer_swht"
xnli,sw,sw,"guaranteed/possible/impossible_swmt"
xnli,sw,sw,"MNLI crowdsource_swmt"
xnli,sw,sw,"GPT-3 style_swmt"
xnli,sw,sw,"justified in saying_swmt"
xnli,sw,sw,"can we infer_swmt"
xnli,vi,vi,"guaranteed/possible/impossible_viht"
xnli,vi,vi,"MNLI crowdsource_viht"
xnli,vi,vi,"GPT-3 style_viht"
xnli,vi,vi,"justified in saying_viht"
xnli,vi,vi,"can we infer_viht"
xnli,vi,vi,"guaranteed/possible/impossible_vimt"
xnli,vi,vi,"MNLI crowdsource_vimt"
xnli,vi,vi,"GPT-3 style_vimt"
xnli,vi,vi,"justified in saying_vimt"
xnli,vi,vi,"can we infer_vimt"
xnli,zh,zh,"guaranteed/possible/impossible_zhht"
xnli,zh,zh,"MNLI crowdsource_zhht"
xnli,zh,zh,"GPT-3 style_zhht"
xnli,zh,zh,"justified in saying_zhht"
xnli,zh,zh,"can we infer_zhht"
xnli,zh,zh,"guaranteed/possible/impossible_zhmt"
xnli,zh,zh,"MNLI crowdsource_zhmt"
xnli,zh,zh,"GPT-3 style_zhmt"
xnli,zh,zh,"justified in saying_zhmt"
xnli,zh,zh,"can we infer_zhmt"
)

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS_VAL[$SLURM_ARRAY_TASK_ID]}
echo $ARGUMENT

# Run T0 evaluation
# For PrefixLM add --prefixlm
IFS=',' read dataset_name dataset_config_name template_config_name template_name split <<< "${DATASET_AND_CONFIG}"
python t-zero/evaluation/run_eval.py \
        --dataset_name $dataset_name \
        --dataset_config_name $dataset_config_name \
        --template_config_name $template_config_name \
        --template_name "$template_name" \
        --split $split \
        --model_name_or_path $CHECKPOINT_PATH \
        --output_dir $OUTPUT_DIR \
        --per_device_eval_batch_size 8 \
        --max_length 2048 \
        --dtype float16
