#!/bin/bash
#SBATCH --job-name=evaluate_mt0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --time 5:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-159

# VALIDATION:
# --array=0-168

# L1
# --array=0-169

# L2
# --array=0-84

# MT L1
# --array=0-69

# MT L2
# --array=0-89

# XNLIMTHT:
# --array=0-79

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate muennighofflmevalgen

CHECKPOINT_PATH=/gpfsssd/scratch/rech/six/commun/experiments/muennighoff/bloomckpt/mt0/mt5-xxl

WORKDIR=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0

pushd $WORKDIR

OUTPUT_DIR=$CHECKPOINT_PATH/evaluation
mkdir -p $OUTPUT_DIR

# Validation
DATASETS_AND_CONFIGS_VAL=(
head_qa,en,en,"multiple_choice_q_and_a_index_with_context_en",validation
head_qa,en,en,"multiple_choice_q_and_a_en",validation
head_qa,en,en,"multiple_choice_q_and_a_index_en",validation
head_qa,en,en,"multiple_choice_a_and_q_with_context_en",validation
head_qa,en,en,"multiple_choice_a_and_q_en",validation
head_qa,es,en,"multiple_choice_q_and_a_index_with_context_en",validation
head_qa,es,en,"multiple_choice_q_and_a_en",validation
head_qa,es,en,"multiple_choice_q_and_a_index_en",validation
head_qa,es,en,"multiple_choice_a_and_q_with_context_en",validation
head_qa,es,en,"multiple_choice_a_and_q_en",validation
climate_fever,None,None,"first_evidence_and_claim_itemization",test
climate_fever,None,None,"claim_and_all_supporting_evidences",test
climate_fever,None,None,"fifth_evidence_and_claim_itemization",test
climate_fever,None,None,"third_evidence_claim_pair",test
climate_fever,None,None,"second_evidence_and_claim_itemization",test
codah,codah,None,"interrogative_instruction_after_sentence_and_choices",train
codah,codah,None,"affirmative_instruction_before_sentence_and_choices",train
codah,codah,None,"affirmative_instruction_after_sentence_and_choices",train
aqua_rat,raw,None,"select_the_best_option",validation
aqua_rat,raw,None,"answer_quiz",validation
aqua_rat,raw,None,"Answer questions from options",validation
commonsense_qa,None,None,"answer_given_question_without_options",validation
commonsense_qa,None,None,"question_answering",validation
commonsense_qa,None,None,"most_suitable_answer",validation
amazon_reviews_multi,en,en,"prompt_title_to_star",validation
amazon_reviews_multi,en,en,"prompt_review_to_star",validation
amazon_reviews_multi,en,en,"prompt_body_title_to_star",validation
amazon_reviews_multi,zh,en,"prompt_title_to_star",validation
amazon_reviews_multi,zh,en,"prompt_review_to_star",validation
amazon_reviews_multi,zh,en,"prompt_body_title_to_star",validation
amazon_reviews_multi,fr,en,"prompt_title_to_star",validation
amazon_reviews_multi,fr,en,"prompt_review_to_star",validation
amazon_reviews_multi,fr,en,"prompt_body_title_to_star",validation
amazon_reviews_multi,es,en,"prompt_title_to_star",validation
amazon_reviews_multi,es,en,"prompt_review_to_star",validation
amazon_reviews_multi,es,en,"prompt_body_title_to_star",validation
art,None,None,"choose_hypothesis_options",validation
art,None,None,"choose_hypothesis_believable",validation
art,None,None,"choose_hypothesis",validation
art,None,None,"choose_hypothesis_desc",validation
art,None,None,"choose_hypothesis_likely",validation
banking77,None,None,"help_page_topic",test
banking77,None,None,"direct_to_which_department",test
banking77,None,None,"rephrase_as_banking_term",test
blbooksgenre,title_genre_classifiction,None,"multi-choice",train
blbooksgenre,title_genre_classifiction,None,"premise_context_first",train
blbooksgenre,title_genre_classifiction,None,"classify",train
blimp,adjunct_island,None,"grammatical_between_1_2",train
blimp,adjunct_island,None,"grammatical_between_A_B",train
blimp,adjunct_island,None,"grammatical_which_one_1_2",train
blimp,adjunct_island,None,"single_sentence_bad_yes_no",train
blimp,adjunct_island,None,"single_sentence_good_yes_no",train
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
liar,None,None,"Given statement guess category",validation
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
pubmed_qa,pqa_labeled,None,"Question Answering (Short)",train
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
selqa,answer_selection_analysis,None,"make-sense-rand",validation
selqa,answer_selection_analysis,None,"which-answer-1st-vs-random",validation
snips_built_in_intents,None,None,"voice_intent",train
snips_built_in_intents,None,None,"categorize_query",train
snips_built_in_intents,None,None,"intent_query",train
snips_built_in_intents,None,None,"categorize_query_brief",train
snips_built_in_intents,None,None,"query_intent",train
)

DATASETS_AND_CONFIGS_L1=(
super_glue,copa,None,"best_option",validation
super_glue,copa,None,"C1 or C2? premise, so/because…",validation
super_glue,copa,None,"i_am_hesitating",validation
super_glue,copa,None,"cause_effect",validation
super_glue,copa,None,"plausible_alternatives",validation
super_glue,rte,None,"MNLI crowdsource",validation
super_glue,rte,None,"GPT-3 style",validation
super_glue,rte,None,"does it follow that",validation
super_glue,rte,None,"should assume",validation
super_glue,rte,None,"guaranteed true",validation
anli,dev_r1,None,"guaranteed/possible/impossible",dev_r1
anli,dev_r1,None,"MNLI crowdsource",dev_r1
anli,dev_r1,None,"GPT-3 style",dev_r1
anli,dev_r1,None,"justified in saying",dev_r1
anli,dev_r1,None,"can we infer",dev_r1
anli,dev_r2,None,"guaranteed/possible/impossible",dev_r2
anli,dev_r2,None,"MNLI crowdsource",dev_r2
anli,dev_r2,None,"GPT-3 style",dev_r2
anli,dev_r2,None,"justified in saying",dev_r2
anli,dev_r2,None,"can we infer",dev_r2
anli,dev_r3,None,"guaranteed/possible/impossible",dev_r3
anli,dev_r3,None,"MNLI crowdsource",dev_r3
anli,dev_r3,None,"GPT-3 style",dev_r3
anli,dev_r3,None,"justified in saying",dev_r3
anli,dev_r3,None,"can we infer",dev_r3
super_glue,cb,None,"guaranteed/possible/impossible",validation
super_glue,cb,None,"MNLI crowdsource",validation
super_glue,cb,None,"GPT-3 style",validation
super_glue,cb,None,"justified in saying",validation
super_glue,cb,None,"can we infer",validation
winogrande,winogrande_xl,None,"underscore refer to",validation
winogrande,winogrande_xl,None,"Replace",validation
winogrande,winogrande_xl,None,"stand for",validation
winogrande,winogrande_xl,None,"does underscore refer to",validation
winogrande,winogrande_xl,None,"True or False",validation
story_cloze,2016,None,"Story Continuation and Options",validation
story_cloze,2016,None,"Answer Given options",validation
story_cloze,2016,None,"Novel Correct Ending",validation
story_cloze,2016,None,"Generate Ending",validation
story_cloze,2016,None,"Choose Story Ending",validation
Muennighoff/xstory_cloze,ar,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,ar,en,"Answer Given options",validation
Muennighoff/xstory_cloze,ar,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,ar,en,"Generate Ending",validation
Muennighoff/xstory_cloze,ar,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,es,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,es,en,"Answer Given options",validation
Muennighoff/xstory_cloze,es,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,es,en,"Generate Ending",validation
Muennighoff/xstory_cloze,es,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,eu,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,eu,en,"Answer Given options",validation
Muennighoff/xstory_cloze,eu,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,eu,en,"Generate Ending",validation
Muennighoff/xstory_cloze,eu,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,id,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,id,en,"Answer Given options",validation
Muennighoff/xstory_cloze,id,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,id,en,"Generate Ending",validation
Muennighoff/xstory_cloze,id,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,hi,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,hi,en,"Answer Given options",validation
Muennighoff/xstory_cloze,hi,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,hi,en,"Generate Ending",validation
Muennighoff/xstory_cloze,hi,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,sw,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,sw,en,"Answer Given options",validation
Muennighoff/xstory_cloze,sw,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,sw,en,"Generate Ending",validation
Muennighoff/xstory_cloze,sw,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,te,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,te,en,"Answer Given options",validation
Muennighoff/xstory_cloze,te,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,te,en,"Generate Ending",validation
Muennighoff/xstory_cloze,te,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,zh,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,zh,en,"Answer Given options",validation
Muennighoff/xstory_cloze,zh,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,zh,en,"Generate Ending",validation
Muennighoff/xstory_cloze,zh,en,"Choose Story Ending",validation
xnli,ar,en,"guaranteed/possible/impossible",validation
xnli,ar,en,"MNLI crowdsource",validation
xnli,ar,en,"GPT-3 style",validation
xnli,ar,en,"justified in saying",validation
xnli,ar,en,"can we infer",validation
xnli,en,en,"guaranteed/possible/impossible",validation
xnli,en,en,"MNLI crowdsource",validation
xnli,en,en,"GPT-3 style",validation
xnli,en,en,"justified in saying",validation
xnli,en,en,"can we infer",validation
xnli,es,en,"guaranteed/possible/impossible",validation
xnli,es,en,"MNLI crowdsource",validation
xnli,es,en,"GPT-3 style",validation
xnli,es,en,"justified in saying",validation
xnli,es,en,"can we infer",validation
xnli,fr,en,"guaranteed/possible/impossible",validation
xnli,fr,en,"MNLI crowdsource",validation
xnli,fr,en,"GPT-3 style",validation
xnli,fr,en,"justified in saying",validation
xnli,fr,en,"can we infer",validation
xnli,hi,en,"guaranteed/possible/impossible",validation
xnli,hi,en,"MNLI crowdsource",validation
xnli,hi,en,"GPT-3 style",validation
xnli,hi,en,"justified in saying",validation
xnli,hi,en,"can we infer",validation
xnli,sw,en,"guaranteed/possible/impossible",validation
xnli,sw,en,"MNLI crowdsource",validation
xnli,sw,en,"GPT-3 style",validation
xnli,sw,en,"justified in saying",validation
xnli,sw,en,"can we infer",validation
xnli,ur,en,"guaranteed/possible/impossible",validation
xnli,ur,en,"MNLI crowdsource",validation
xnli,ur,en,"GPT-3 style",validation
xnli,ur,en,"justified in saying",validation
xnli,ur,en,"can we infer",validation
xnli,vi,en,"guaranteed/possible/impossible",validation
xnli,vi,en,"MNLI crowdsource",validation
xnli,vi,en,"GPT-3 style",validation
xnli,vi,en,"justified in saying",validation
xnli,vi,en,"can we infer",validation
xnli,zh,en,"guaranteed/possible/impossible",validation
xnli,zh,en,"MNLI crowdsource",validation
xnli,zh,en,"GPT-3 style",validation
xnli,zh,en,"justified in saying",validation
xnli,zh,en,"can we infer",validation
xcopa,id,en,"best_option",validation
xcopa,id,en,"C1 or C2? premise, so/because…",validation
xcopa,id,en,"i_am_hesitating",validation
xcopa,id,en,"cause_effect",validation
xcopa,id,en,"plausible_alternatives",validation
xcopa,sw,en,"best_option",validation
xcopa,sw,en,"C1 or C2? premise, so/because…",validation
xcopa,sw,en,"i_am_hesitating",validation
xcopa,sw,en,"cause_effect",validation
xcopa,sw,en,"plausible_alternatives",validation
xcopa,ta,en,"best_option",validation
xcopa,ta,en,"C1 or C2? premise, so/because…",validation
xcopa,ta,en,"i_am_hesitating",validation
xcopa,ta,en,"cause_effect",validation
xcopa,ta,en,"plausible_alternatives",validation
xcopa,vi,en,"best_option",validation
xcopa,vi,en,"C1 or C2? premise, so/because…",validation
xcopa,vi,en,"i_am_hesitating",validation
xcopa,vi,en,"cause_effect",validation
xcopa,vi,en,"plausible_alternatives",validation
xcopa,zh,en,"best_option",validation
xcopa,zh,en,"C1 or C2? premise, so/because…",validation
xcopa,zh,en,"i_am_hesitating",validation
xcopa,zh,en,"cause_effect",validation
xcopa,zh,en,"plausible_alternatives",validation
Muennighoff/xwinograd,en,en,"underscore refer to",test
Muennighoff/xwinograd,en,en,"Replace",test
Muennighoff/xwinograd,en,en,"stand for",test
Muennighoff/xwinograd,en,en,"does underscore refer to",test
Muennighoff/xwinograd,en,en,"True or False",test
Muennighoff/xwinograd,fr,en,"underscore refer to",test
Muennighoff/xwinograd,fr,en,"Replace",test
Muennighoff/xwinograd,fr,en,"stand for",test
Muennighoff/xwinograd,fr,en,"does underscore refer to",test
Muennighoff/xwinograd,fr,en,"True or False",test
Muennighoff/xwinograd,pt,en,"underscore refer to",test
Muennighoff/xwinograd,pt,en,"Replace",test
Muennighoff/xwinograd,pt,en,"stand for",test
Muennighoff/xwinograd,pt,en,"does underscore refer to",test
Muennighoff/xwinograd,pt,en,"True or False",test
Muennighoff/xwinograd,zh,en,"underscore refer to",test
Muennighoff/xwinograd,zh,en,"Replace",test
Muennighoff/xwinograd,zh,en,"stand for",test
Muennighoff/xwinograd,zh,en,"does underscore refer to",test
Muennighoff/xwinograd,zh,en,"True or False",test
)

DATASETS_AND_CONFIGS_L2=(
Muennighoff/xstory_cloze,ru,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,ru,en,"Answer Given options",validation
Muennighoff/xstory_cloze,ru,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,ru,en,"Generate Ending",validation
Muennighoff/xstory_cloze,ru,en,"Choose Story Ending",validation
Muennighoff/xstory_cloze,my,en,"Story Continuation and Options",validation
Muennighoff/xstory_cloze,my,en,"Answer Given options",validation
Muennighoff/xstory_cloze,my,en,"Novel Correct Ending",validation
Muennighoff/xstory_cloze,my,en,"Generate Ending",validation
Muennighoff/xstory_cloze,my,en,"Choose Story Ending",validation
xnli,bg,en,"guaranteed/possible/impossible",validation
xnli,bg,en,"MNLI crowdsource",validation
xnli,bg,en,"GPT-3 style",validation
xnli,bg,en,"justified in saying",validation
xnli,bg,en,"can we infer",validation
xnli,de,en,"guaranteed/possible/impossible",validation
xnli,de,en,"MNLI crowdsource",validation
xnli,de,en,"GPT-3 style",validation
xnli,de,en,"justified in saying",validation
xnli,de,en,"can we infer",validation
xnli,el,en,"guaranteed/possible/impossible",validation
xnli,el,en,"MNLI crowdsource",validation
xnli,el,en,"GPT-3 style",validation
xnli,el,en,"justified in saying",validation
xnli,el,en,"can we infer",validation
xnli,ru,en,"guaranteed/possible/impossible",validation
xnli,ru,en,"MNLI crowdsource",validation
xnli,ru,en,"GPT-3 style",validation
xnli,ru,en,"justified in saying",validation
xnli,ru,en,"can we infer",validation
xnli,th,en,"guaranteed/possible/impossible",validation
xnli,th,en,"MNLI crowdsource",validation
xnli,th,en,"GPT-3 style",validation
xnli,th,en,"justified in saying",validation
xnli,th,en,"can we infer",validation
xnli,tr,en,"guaranteed/possible/impossible",validation
xnli,tr,en,"MNLI crowdsource",validation
xnli,tr,en,"GPT-3 style",validation
xnli,tr,en,"justified in saying",validation
xnli,tr,en,"can we infer",validation
Muennighoff/xwinograd,ru,en,"underscore refer to",test
Muennighoff/xwinograd,ru,en,"Replace",test
Muennighoff/xwinograd,ru,en,"stand for",test
Muennighoff/xwinograd,ru,en,"does underscore refer to",test
Muennighoff/xwinograd,ru,en,"True or False",test
Muennighoff/xwinograd,jp,en,"underscore refer to",test
Muennighoff/xwinograd,jp,en,"Replace",test
Muennighoff/xwinograd,jp,en,"stand for",test
Muennighoff/xwinograd,jp,en,"does underscore refer to",test
Muennighoff/xwinograd,jp,en,"True or False",test
xcopa,et,en,"best_option",validation
xcopa,et,en,"C1 or C2? premise, so/because…",validation
xcopa,et,en,"i_am_hesitating",validation
xcopa,et,en,"cause_effect",validation
xcopa,et,en,"plausible_alternatives",validation
xcopa,ht,en,"best_option",validation
xcopa,ht,en,"C1 or C2? premise, so/because…",validation
xcopa,ht,en,"i_am_hesitating",validation
xcopa,ht,en,"cause_effect",validation
xcopa,ht,en,"plausible_alternatives",validation
xcopa,it,en,"best_option",validation
xcopa,it,en,"C1 or C2? premise, so/because…",validation
xcopa,it,en,"i_am_hesitating",validation
xcopa,it,en,"cause_effect",validation
xcopa,it,en,"plausible_alternatives",validation
xcopa,qu,en,"best_option",validation
xcopa,qu,en,"C1 or C2? premise, so/because…",validation
xcopa,qu,en,"i_am_hesitating",validation
xcopa,qu,en,"cause_effect",validation
xcopa,qu,en,"plausible_alternatives",validation
xcopa,th,en,"best_option",validation
xcopa,th,en,"C1 or C2? premise, so/because…",validation
xcopa,th,en,"i_am_hesitating",validation
xcopa,th,en,"cause_effect",validation
xcopa,th,en,"plausible_alternatives",validation
xcopa,tr,en,"best_option",validation
xcopa,tr,en,"C1 or C2? premise, so/because…",validation
xcopa,tr,en,"i_am_hesitating",validation
xcopa,tr,en,"cause_effect",validation
xcopa,tr,en,"plausible_alternatives",validation
)

DATASETS_AND_CONFIGS_MT_L1=(
Muennighoff/xstory_cloze,ar,ar,"Story Continuation and Options_armt",validation
Muennighoff/xstory_cloze,ar,ar,"Answer Given options_armt",validation
Muennighoff/xstory_cloze,ar,ar,"Novel Correct Ending_armt",validation
Muennighoff/xstory_cloze,ar,ar,"Generate Ending_armt",validation
Muennighoff/xstory_cloze,ar,ar,"Choose Story Ending_armt",validation
Muennighoff/xstory_cloze,es,es,"Story Continuation and Options_esmt",validation
Muennighoff/xstory_cloze,es,es,"Answer Given options_esmt",validation
Muennighoff/xstory_cloze,es,es,"Novel Correct Ending_esmt",validation
Muennighoff/xstory_cloze,es,es,"Generate Ending_esmt",validation
Muennighoff/xstory_cloze,es,es,"Choose Story Ending_esmt",validation
Muennighoff/xstory_cloze,eu,eu,"Story Continuation and Options_eumt",validation
Muennighoff/xstory_cloze,eu,eu,"Answer Given options_eumt",validation
Muennighoff/xstory_cloze,eu,eu,"Novel Correct Ending_eumt",validation
Muennighoff/xstory_cloze,eu,eu,"Generate Ending_eumt",validation
Muennighoff/xstory_cloze,eu,eu,"Choose Story Ending_eumt",validation
Muennighoff/xstory_cloze,id,id,"Story Continuation and Options_idmt",validation
Muennighoff/xstory_cloze,id,id,"Answer Given options_idmt",validation
Muennighoff/xstory_cloze,id,id,"Novel Correct Ending_idmt",validation
Muennighoff/xstory_cloze,id,id,"Generate Ending_idmt",validation
Muennighoff/xstory_cloze,id,id,"Choose Story Ending_idmt",validation
Muennighoff/xstory_cloze,hi,hi,"Story Continuation and Options_himt",validation
Muennighoff/xstory_cloze,hi,hi,"Answer Given options_himt",validation
Muennighoff/xstory_cloze,hi,hi,"Novel Correct Ending_himt",validation
Muennighoff/xstory_cloze,hi,hi,"Generate Ending_himt",validation
Muennighoff/xstory_cloze,hi,hi,"Choose Story Ending_himt",validation
Muennighoff/xstory_cloze,sw,sw,"Story Continuation and Options_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Answer Given options_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Novel Correct Ending_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Generate Ending_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Choose Story Ending_swmt",validation
Muennighoff/xstory_cloze,te,te,"Story Continuation and Options_temt",validation
Muennighoff/xstory_cloze,te,te,"Answer Given options_temt",validation
Muennighoff/xstory_cloze,te,te,"Novel Correct Ending_temt",validation
Muennighoff/xstory_cloze,te,te,"Generate Ending_temt",validation
Muennighoff/xstory_cloze,te,te,"Choose Story Ending_temt",validation
Muennighoff/xstory_cloze,zh,zh,"Story Continuation and Options_zhmt",validation
Muennighoff/xstory_cloze,zh,zh,"Answer Given options_zhmt",validation
Muennighoff/xstory_cloze,zh,zh,"Novel Correct Ending_zhmt",validation
Muennighoff/xstory_cloze,zh,zh,"Generate Ending_zhmt",validation
Muennighoff/xstory_cloze,zh,zh,"Choose Story Ending_zhmt",validation
Muennighoff/xwinograd,fr,fr,"underscore refer to_frmt",test
Muennighoff/xwinograd,fr,fr,"Replace_frmt",test
Muennighoff/xwinograd,fr,fr,"stand for_frmt",test
Muennighoff/xwinograd,fr,fr,"does underscore refer to_frmt",test
Muennighoff/xwinograd,fr,fr,"True or False_frmt",test
Muennighoff/xwinograd,pt,pt,"underscore refer to_ptmt",test
Muennighoff/xwinograd,pt,pt,"Replace_ptmt",test
Muennighoff/xwinograd,pt,pt,"stand for_ptmt",test
Muennighoff/xwinograd,pt,pt,"does underscore refer to_ptmt",test
Muennighoff/xwinograd,pt,pt,"True or False_ptmt",test
Muennighoff/xwinograd,zh,zh,"underscore refer to_zhmt",test
Muennighoff/xwinograd,zh,zh,"Replace_zhmt",test
Muennighoff/xwinograd,zh,zh,"stand for_zhmt",test
Muennighoff/xwinograd,zh,zh,"does underscore refer to_zhmt",test
Muennighoff/xwinograd,zh,zh,"True or False_zhmt",test
xcopa,id,id,"best_option_idmt",validation
xcopa,id,id,"C1 or C2? premise_idmt",validation
xcopa,id,id,"i_am_hesitating_idmt",validation
xcopa,id,id,"cause_effect_idmt",validation
xcopa,id,id,"plausible_alternatives_idmt",validation
xcopa,sw,sw,"best_option_swmt",validation
xcopa,sw,sw,"C1 or C2? premise_swmt",validation
xcopa,sw,sw,"i_am_hesitating_swmt",validation
xcopa,sw,sw,"cause_effect_swmt",validation
xcopa,sw,sw,"plausible_alternatives_swmt",validation
xcopa,ta,ta,"best_option_tamt",validation
xcopa,ta,ta,"C1 or C2? premise_tamt",validation
xcopa,ta,ta,"i_am_hesitating_tamt",validation
xcopa,ta,ta,"cause_effect_tamt",validation
xcopa,ta,ta,"plausible_alternatives_tamt",validation
xcopa,vi,vi,"best_option_vimt",validation
xcopa,vi,vi,"C1 or C2? premise_vimt",validation
xcopa,vi,vi,"i_am_hesitating_vimt",validation
xcopa,vi,vi,"cause_effect_vimt",validation
xcopa,vi,vi,"plausible_alternatives_vimt",validation
xcopa,zh,zh,"best_option_zhmt",validation
xcopa,zh,zh,"C1 or C2? premise_zhmt",validation
xcopa,zh,zh,"i_am_hesitating_zhmt",validation
xcopa,zh,zh,"cause_effect_zhmt",validation
xcopa,zh,zh,"plausible_alternatives_zhmt",validation   
)

DATASETS_AND_CONFIGS_ZHHT=(
Muennighoff/xstory_cloze,zh,zh,"Story Continuation and Options_zhht",validation
Muennighoff/xstory_cloze,zh,zh,"Answer Given options_zhht",validation
Muennighoff/xstory_cloze,zh,zh,"Novel Correct Ending_zhht",validation
Muennighoff/xstory_cloze,zh,zh,"Generate Ending_zhht",validation
Muennighoff/xstory_cloze,zh,zh,"Choose Story Ending_zhht",validation
Muennighoff/xwinograd,zh,zh,"underscore refer to_zhht",test
Muennighoff/xwinograd,zh,zh,"Replace_zhht",test
Muennighoff/xwinograd,zh,zh,"stand for_zhht",test
Muennighoff/xwinograd,zh,zh,"does underscore refer to_zhht",test
Muennighoff/xwinograd,zh,zh,"True or False_zhht",test
xcopa,zh,zh,"best_option_zhht",validation
xcopa,zh,zh,"C1 or C2? premise_zhht",validation
xcopa,zh,zh,"i_am_hesitating_zhht",validation
xcopa,zh,zh,"cause_effect_zhht",validation
xcopa,zh,zh,"plausible_alternatives_zhht",validation  
)

DATASETS_AND_CONFIGS_XNLIHTMT=(
xnli,ar,ar,"guaranteed/possible/impossible_arht",validation
xnli,ar,ar,"MNLI crowdsource_arht",validation
xnli,ar,ar,"GPT-3 style_arht",validation
xnli,ar,ar,"justified in saying_arht",validation
xnli,ar,ar,"can we infer_arht",validation
xnli,ar,ar,"guaranteed/possible/impossible_armt",validation
xnli,ar,ar,"MNLI crowdsource_armt",validation
xnli,ar,ar,"GPT-3 style_armt",validation
xnli,ar,ar,"justified in saying_armt",validation
xnli,ar,ar,"can we infer_armt",validation
xnli,es,es,"guaranteed/possible/impossible_esht",validation
xnli,es,es,"MNLI crowdsource_esht",validation
xnli,es,es,"GPT-3 style_esht",validation
xnli,es,es,"justified in saying_esht",validation
xnli,es,es,"can we infer_esht",validation
xnli,es,es,"guaranteed/possible/impossible_esmt",validation
xnli,es,es,"MNLI crowdsource_esmt",validation
xnli,es,es,"GPT-3 style_esmt",validation
xnli,es,es,"justified in saying_esmt",validation
xnli,es,es,"can we infer_esmt",validation
xnli,fr,fr,"guaranteed/possible/impossible_frht",validation
xnli,fr,fr,"MNLI crowdsource_frht",validation
xnli,fr,fr,"GPT-3 style_frht",validation
xnli,fr,fr,"justified in saying_frht",validation
xnli,fr,fr,"can we infer_frht",validation
xnli,fr,fr,"guaranteed/possible/impossible_frmt",validation
xnli,fr,fr,"MNLI crowdsource_frmt",validation
xnli,fr,fr,"GPT-3 style_frmt",validation
xnli,fr,fr,"justified in saying_frmt",validation
xnli,fr,fr,"can we infer_frmt",validation
xnli,hi,hi,"guaranteed/possible/impossible_hiht",validation
xnli,hi,hi,"MNLI crowdsource_hiht",validation
xnli,hi,hi,"GPT-3 style_hiht",validation
xnli,hi,hi,"justified in saying_hiht",validation
xnli,hi,hi,"can we infer_hiht",validation
xnli,hi,hi,"guaranteed/possible/impossible_himt",validation
xnli,hi,hi,"MNLI crowdsource_himt",validation
xnli,hi,hi,"GPT-3 style_himt",validation
xnli,hi,hi,"justified in saying_himt",validation
xnli,hi,hi,"can we infer_himt",validation
xnli,ur,ur,"guaranteed/possible/impossible_urht",validation
xnli,ur,ur,"MNLI crowdsource_urht",validation
xnli,ur,ur,"GPT-3 style_urht",validation
xnli,ur,ur,"justified in saying_urht",validation
xnli,ur,ur,"can we infer_urht",validation
xnli,ur,ur,"guaranteed/possible/impossible_urmt",validation
xnli,ur,ur,"MNLI crowdsource_urmt",validation
xnli,ur,ur,"GPT-3 style_urmt",validation
xnli,ur,ur,"justified in saying_urmt",validation
xnli,ur,ur,"can we infer_urmt",validation
xnli,sw,sw,"guaranteed/possible/impossible_swht",validation
xnli,sw,sw,"MNLI crowdsource_swht",validation
xnli,sw,sw,"GPT-3 style_swht",validation
xnli,sw,sw,"justified in saying_swht",validation
xnli,sw,sw,"can we infer_swht",validation
xnli,sw,sw,"guaranteed/possible/impossible_swmt",validation
xnli,sw,sw,"MNLI crowdsource_swmt",validation
xnli,sw,sw,"GPT-3 style_swmt",validation
xnli,sw,sw,"justified in saying_swmt",validation
xnli,sw,sw,"can we infer_swmt",validation
xnli,vi,vi,"guaranteed/possible/impossible_viht",validation
xnli,vi,vi,"MNLI crowdsource_viht",validation
xnli,vi,vi,"GPT-3 style_viht",validation
xnli,vi,vi,"justified in saying_viht",validation
xnli,vi,vi,"can we infer_viht",validation
xnli,vi,vi,"guaranteed/possible/impossible_vimt",validation
xnli,vi,vi,"MNLI crowdsource_vimt",validation
xnli,vi,vi,"GPT-3 style_vimt",validation
xnli,vi,vi,"justified in saying_vimt",validation
xnli,vi,vi,"can we infer_vimt",validation
xnli,zh,zh,"guaranteed/possible/impossible_zhht",validation
xnli,zh,zh,"MNLI crowdsource_zhht",validation
xnli,zh,zh,"GPT-3 style_zhht",validation
xnli,zh,zh,"justified in saying_zhht",validation
xnli,zh,zh,"can we infer_zhht",validation
xnli,zh,zh,"guaranteed/possible/impossible_zhmt",validation
xnli,zh,zh,"MNLI crowdsource_zhmt",validation
xnli,zh,zh,"GPT-3 style_zhmt",validation
xnli,zh,zh,"justified in saying_zhmt",validation
xnli,zh,zh,"can we infer_zhmt",validation
)

DATASETS_AND_CONFIGS_MT_L2=(
Muennighoff/xstory_cloze,my,my,"Story Continuation and Options_mymt",validation
Muennighoff/xstory_cloze,my,my,"Answer Given options_mymt",validation
Muennighoff/xstory_cloze,my,my,"Novel Correct Ending_mymt",validation
Muennighoff/xstory_cloze,my,my,"Generate Ending_mymt",validation
Muennighoff/xstory_cloze,my,my,"Choose Story Ending_mymt",validation
Muennighoff/xstory_cloze,ru,ru,"Story Continuation and Options_rumt",validation
Muennighoff/xstory_cloze,ru,ru,"Answer Given options_rumt",validation
Muennighoff/xstory_cloze,ru,ru,"Novel Correct Ending_rumt",validation
Muennighoff/xstory_cloze,ru,ru,"Generate Ending_rumt",validation
Muennighoff/xstory_cloze,ru,ru,"Choose Story Ending_rumt",validation
Muennighoff/xstory_cloze,sw,sw,"Story Continuation and Options_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Answer Given options_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Novel Correct Ending_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Generate Ending_swmt",validation
Muennighoff/xstory_cloze,sw,sw,"Choose Story Ending_swmt",validation
Muennighoff/xstory_cloze,te,te,"Story Continuation and Options_temt",validation
Muennighoff/xstory_cloze,te,te,"Answer Given options_temt",validation
Muennighoff/xstory_cloze,te,te,"Novel Correct Ending_temt",validation
Muennighoff/xstory_cloze,te,te,"Generate Ending_temt",validation
Muennighoff/xstory_cloze,te,te,"Choose Story Ending_temt",validation
Muennighoff/xwinograd,jp,jp,"underscore refer to_jpmt",test
Muennighoff/xwinograd,jp,jp,"Replace_jpmt",test
Muennighoff/xwinograd,jp,jp,"stand for_jpmt",test
Muennighoff/xwinograd,jp,jp,"does underscore refer to_jpmt",test
Muennighoff/xwinograd,jp,jp,"True or False_jpmt",test
Muennighoff/xwinograd,ru,ru,"underscore refer to_rumt",test
Muennighoff/xwinograd,ru,ru,"Replace_rumt",test
Muennighoff/xwinograd,ru,ru,"stand for_rumt",test
Muennighoff/xwinograd,ru,ru,"does underscore refer to_rumt",test
Muennighoff/xwinograd,ru,ru,"True or False_rumt",test
xcopa,et,et,"best_option_etmt",validation
xcopa,et,et,"C1 or C2? premise_etmt",validation
xcopa,et,et,"i_am_hesitating_etmt",validation
xcopa,et,et,"cause_effect_etmt",validation
xcopa,et,et,"plausible_alternatives_etmt",validation
xcopa,ht,ht,"best_option_htmt",validation
xcopa,ht,ht,"C1 or C2? premise_htmt",validation
xcopa,ht,ht,"i_am_hesitating_htmt",validation
xcopa,ht,ht,"cause_effect_htmt",validation
xcopa,ht,ht,"plausible_alternatives_htmt",validation
xcopa,it,it,"best_option_itmt",validation
xcopa,it,it,"C1 or C2? premise_itmt",validation
xcopa,it,it,"i_am_hesitating_itmt",validation
xcopa,it,it,"cause_effect_itmt",validation
xcopa,it,it,"plausible_alternatives_itmt",validation
xcopa,qu,qu,"best_option_qumt",validation
xcopa,qu,qu,"C1 or C2? premise_qumt",validation
xcopa,qu,qu,"i_am_hesitating_qumt",validation
xcopa,qu,qu,"cause_effect_qumt",validation
xcopa,qu,qu,"plausible_alternatives_qumt",validation
xcopa,th,th,"best_option_thmt",validation
xcopa,th,th,"C1 or C2? premise_thmt",validation
xcopa,th,th,"i_am_hesitating_thmt",validation
xcopa,th,th,"cause_effect_thmt",validation
xcopa,th,th,"plausible_alternatives_thmt",validation
xcopa,tr,tr,"best_option_trmt",validation
xcopa,tr,tr,"C1 or C2? premise_trmt",validation
xcopa,tr,tr,"i_am_hesitating_trmt",validation
xcopa,tr,tr,"cause_effect_trmt",validation
xcopa,tr,tr,"plausible_alternatives_trmt",validation
xnli,bg,bg,"guaranteed/possible/impossible_bgmt",validation
xnli,bg,bg,"MNLI crowdsource_bgmt",validation
xnli,bg,bg,"GPT-3 style_bgmt",validation
xnli,bg,bg,"justified in saying_bgmt",validation
xnli,bg,bg,"can we infer_bgmt",validation
xnli,de,de,"guaranteed/possible/impossible_demt",validation
xnli,de,de,"MNLI crowdsource_demt",validation
xnli,de,de,"GPT-3 style_demt",validation
xnli,de,de,"justified in saying_demt",validation
xnli,de,de,"can we infer_demt",validation
xnli,el,el,"guaranteed/possible/impossible_elmt",validation
xnli,el,el,"MNLI crowdsource_elmt",validation
xnli,el,el,"GPT-3 style_elmt",validation
xnli,el,el,"justified in saying_elmt",validation
xnli,el,el,"can we infer_elmt",validation
xnli,ru,ru,"guaranteed/possible/impossible_rumt",validation
xnli,ru,ru,"MNLI crowdsource_rumt",validation
xnli,ru,ru,"GPT-3 style_rumt",validation
xnli,ru,ru,"justified in saying_rumt",validation
xnli,ru,ru,"can we infer_rumt",validation
xnli,th,th,"guaranteed/possible/impossible_thmt",validation
xnli,th,th,"MNLI crowdsource_thmt",validation
xnli,th,th,"GPT-3 style_thmt",validation
xnli,th,th,"justified in saying_thmt",validation
xnli,th,th,"can we infer_thmt",validation
xnli,tr,tr,"guaranteed/possible/impossible_trmt",validation
xnli,tr,tr,"MNLI crowdsource_trmt",validation
xnli,tr,tr,"GPT-3 style_trmt",validation
xnli,tr,tr,"justified in saying_trmt",validation
xnli,tr,tr,"can we infer_trmt",validation
)

DATASET_AND_CONFIG=${DATASETS_AND_CONFIGS_L1[$SLURM_ARRAY_TASK_ID]}
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
        --per_device_eval_batch_size 4 \
        --max_length 512 \
        --dtype float32 \
        --nospace
