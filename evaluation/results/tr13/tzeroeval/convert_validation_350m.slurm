#!/bin/bash
#SBATCH --job-name=ckpts
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --account=ajs@cpu
#SBATCH --partition=cpu_p1

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
conda activate muennighoffmodelconv

CKPT_PATH=$six_ALL_CCFRSCRATCH/checkpoints/tr13e-350M-ml-t0/checkpoints/xp3capmixnewcodelonglossseq

CKPTS=(
global_step250
global_step500
global_step750
global_step1000
global_step1250
global_step1500
global_step1750
global_step2000
global_step2250
global_step2500
global_step2750
global_step3000
)
EXAMPLE_CKPT=/gpfsssd/scratch/rech/six/commun/experiments/muennighoff/bloomckpt/350m/bloom-560m
DUMP_PATH=$six_ALL_CCFRSCRATCH/commun/experiments/muennighoff/bloomckpt/350mt0
OUT_PREFIX=xp3capmixnewcodelonglossseq

TP=1

### CONVERT ###


for i in {0..12}; do
CKPT=${CKPTS[$i]}
echo "$i"
echo "Running $CKPT"

OUTPUTCKPT=$DUMP_PATH/"$OUT_PREFIX$CKPT"
python $six_ALL_CCFRSCRATCH/commun/experiments/muennighoff/bloomckpt/transformers_clone/src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py --pytorch_dump_folder_path $OUTPUTCKPT --bloom_checkpoint_path $CKPT_PATH/$CKPT --pretraining_tp $TP --bloom_config_file $EXAMPLE_CKPT/config.json

# Copy tokenizer.json etc
cp -r $EXAMPLE_CKPT/*.json $OUTPUTCKPT/

eval_script="./eval_$i.slurm"
cat <<EOT > $eval_script
#!/bin/bash
#SBATCH --job-name=evaluate_t0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --time 5:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-168

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate muennighofflmevalgen

CHECKPOINT_PATH=$OUTPUTCKPT

WORKDIR=/gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0
pushd "\$WORKDIR"
OUTPUT_DIR="\$CHECKPOINT_PATH/evaluation"
mkdir -p "\$OUTPUT_DIR"

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

DATASET_AND_CONFIG="\${DATASETS_AND_CONFIGS_VAL[\$SLURM_ARRAY_TASK_ID]}"
echo "\$ARGUMENT"

# Run T0 evaluation
# For PrefixLM add --prefixlm
IFS=',' read dataset_name dataset_config_name template_config_name template_name split <<< "\${DATASET_AND_CONFIG}"
python t-zero/evaluation/run_eval.py \
        --dataset_name "\$dataset_name" \
        --dataset_config_name "\$dataset_config_name" \
        --template_config_name "\$template_config_name" \
        --template_name "\$template_name" \
        --split "\$split" \
        --model_name_or_path "\$CHECKPOINT_PATH" \
        --output_dir "\$OUTPUT_DIR" \
        --per_device_eval_batch_size 4 \
        --max_length 2048 \
        --dtype float16
EOT

sbatch $eval_script


lm_eval_script="./lm_eval_$i.slurm"
cat <<EOT > $lm_eval_script
#!/bin/bash
#SBATCH --job-name=lmeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --time 20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-22

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
conda activate muennighofflmevalgen

echo "START TIME: $(date)"

# defining the right environment variables
export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasets
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Converted transformer checkpoint
MODEL_CKPT=$OUTPUTCKPT

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/lm-evaluation-harness


DATASETS_AND_CONFIGS=(
wmt14_fr_en,fr-en,"version-en-fr-target"
wmt14_fr_en,fr-en,"a_good_translation-en-fr-target"
wmt14_fr_en,fr-en,"a_good_translation-en-fr-source+target"
wmt14_fr_en,fr-en,"xglm-en-fr-target"
wmt14_fr_en,fr-en,"gpt3-en-fr"
wmt14_fr_en,fr-en,"version-fr-en-target"
wmt14_fr_en,fr-en,"a_good_translation-fr-en-target"
wmt14_fr_en,fr-en,"a_good_translation-fr-en-source+target"
wmt14_fr_en,fr-en,"xglm-fr-en-target"
wmt14_fr_en,fr-en,"gpt3-fr-en"
wmt14_hi_en,hi-en,"version-en-hi-target"
wmt14_hi_en,hi-en,"a_good_translation-en-hi-target"
wmt14_hi_en,hi-en,"a_good_translation-en-hi-source+target"
wmt14_hi_en,hi-en,"xglm-en-hi-target"
wmt14_hi_en,hi-en,"gpt-3-en-hi-target"
wmt14_hi_en,hi-en,"version-hi-en-target"
wmt14_hi_en,hi-en,"a_good_translation-hi-en-target"
wmt14_hi_en,hi-en,"a_good_translation-hi-en-source+target"
wmt14_hi_en,hi-en,"xglm-hi-en-target"
wmt14_hi_en,hi-en,"gpt-3-hi-en-target"
mlsum_es,"es","layman_summ_es"
mlsum_es,"es","palm_prompt"
mlsum_es,"es","summarise_this_in_es_few_sentences"
)

DATASET_AND_CONFIG="\${DATASETS_AND_CONFIGS[\$SLURM_ARRAY_TASK_ID]}"
echo "\$ARGUMENT"

IFS=',' read dataset_name lang template_name <<< "\${DATASET_AND_CONFIG}"

# Use this fork of lm-eval: https://github.com/bigscience-workshop/lm-evaluation-harness/pull/109
python main.py \
    --model_api_name 'hf-causal' \
    --model_args "pretrained=\$MODEL_CKPT,use_accelerate=True,tokenizer=\$MODEL_CKPT,dtype=float16" \
    --device cuda \
    --batch_size 16 \
    --no_tracking \
    --task_name "\$dataset_name" \
    --template_names "\$template_name" \
    --bootstrap_iters 10 \
    --limit 3000

mkdir -p "$OUTPUTCKPT/evaluation/\$dataset_name"
mv "outputs/*$CKPT*\$dataset_name*" "$OUTPUTCKPT/evaluation/\$dataset_name/"

echo "END TIME: $(date)"
EOT

sbatch $lm_eval_script


done
