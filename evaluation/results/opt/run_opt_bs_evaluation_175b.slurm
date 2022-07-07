#!/bin/bash
#SBATCH --job-name=bs-eval-opt-175b
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/six/uty16tp/logs/%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --array=0-171

set -x -e

source $six_ALL_CCFRWORK/start-py38-pt111
conda activate muennighofflmeval

echo "START TIME: $(date)"


CHECKPOINT_PATH=$six_ALL_CCFRSCRATCH/opt/opt-175b-meg-ds
MEGATRON_DEEPSPEED_REPO=/gpfswork/rech/six/uty16tp/code/big_science/Megatron-DeepSpeed
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRANSFORMERS_CACHE=$six_ALL_CCFRWORK/models
export HF_DATASETS_CACHE=$six_ALL_CCFRWORK/datasetseval
export HF_MODULES_CACHE=$six_ALL_CCFRWORK/modules
export HF_METRICS_CACHE=$six_ALL_CCFRWORK/metrics

cd $MEGATRON_DEEPSPEED_REPO

# Make sure you use the slow version of the tokenizer.
# Same tokenizer for 125m and 175b
TOKENIZER_NAME_OR_PATH=/gpfsscratch/rech/six/commun/opt/opt-125m

PP_SIZE=8
TP_SIZE=1

NHIDDEN=12288
NLAYERS=96
NHEADS=96
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2050

# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1

MEGATRON_REQUIRED_ARGS="
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --fp16 \
    --relu \
    --seed 42 \
    --pad-vocab-size-to 50272 \
    --make-vocab-size-divisible-by 1\
    --no-bias-gelu-fusion\
"


ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 1,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

# --task_list GEM/web_nlg_en,GEM/web_nlg_en_challenge_test_numbers,GEM/web_nlg_en_challenge_test_scramble,GEM/web_nlg_en_challenge_validation_sample,GEM/web_nlg_ru,GEM/web_nlg_ru_challenge_test_scramble,GEM/web_nlg_ru_challenge_validation_sample,GEM/wiki_auto_asset_turk_challenge_test_asset_backtranslation,GEM/wiki_auto_asset_turk_challenge_test_asset_bfp02,GEM/wiki_auto_asset_turk_challenge_test_asset_bfp05,GEM/wiki_auto_asset_turk_challenge_test_asset_nopunc,GEM/wiki_auto_asset_turk_challenge_test_turk_backtranslation,GEM/wiki_auto_asset_turk_challenge_test_turk_bfp02,GEM/wiki_auto_asset_turk_challenge_test_turk_bfp05,GEM/wiki_auto_asset_turk_challenge_test_turk_nopunc,GEM/wiki_auto_asset_turk_test_asset,GEM/wiki_auto_asset_turk_test_turk,GEM/wiki_lingua_ar,GEM/wiki_lingua_cs,GEM/wiki_lingua_de,GEM/wiki_lingua_en,GEM/wiki_lingua_es,GEM/wiki_lingua_fr,GEM/wiki_lingua_hi,GEM/wiki_lingua_id,GEM/wiki_lingua_it,GEM/wiki_lingua_ja,GEM/wiki_lingua_ko,GEM/wiki_lingua_nl,GEM/wiki_lingua_pt,GEM/wiki_lingua_ru,GEM/wiki_lingua_th,GEM/wiki_lingua_tr,GEM/wiki_lingua_vi,GEM/wiki_lingua_zh,gem_xsum,gem_xsum_challenge_sample,gem_xsum_challenge_test_backtranslation,gem_xsum_challenge_test_bfp_02,gem_xsum_challenge_test_bfp_05,gem_xsum_challenge_test_covid,gem_xsum_challenge_test_nopunc,axb,axg,boolq,cb,cola,copa,crows_pairs_english,crows_pairs_french,diabla,e2e_nlg_cleaned,mnli,mnli_mismatched,multirc,piaf,qqp,rte,sst,tydiqa_primary,tydiqa_secondary,wic,wsc,wnli,wino_bias_type1_anti,wino_bias_type1_pro,wino_bias_type2_anti,wino_bias_type2_pro,xquad_ar,xquad_en\

TASKS=(
GEM/web_nlg_en
GEM/web_nlg_en_challenge_test_numbers
GEM/web_nlg_en_challenge_test_scramble
GEM/web_nlg_en_challenge_validation_sample
GEM/web_nlg_ru
GEM/web_nlg_ru_challenge_test_scramble
GEM/web_nlg_ru_challenge_validation_sample
GEM/wiki_auto_asset_turk_challenge_test_asset_backtranslation
GEM/wiki_auto_asset_turk_challenge_test_asset_bfp02
GEM/wiki_auto_asset_turk_challenge_test_asset_bfp05
GEM/wiki_auto_asset_turk_challenge_test_asset_nopunc
GEM/wiki_auto_asset_turk_challenge_test_turk_backtranslation
GEM/wiki_auto_asset_turk_challenge_test_turk_bfp02
GEM/wiki_auto_asset_turk_challenge_test_turk_bfp05
GEM/wiki_auto_asset_turk_challenge_test_turk_nopunc
GEM/wiki_auto_asset_turk_test_asset
GEM/wiki_auto_asset_turk_test_turk
GEM/wiki_lingua_ar
GEM/wiki_lingua_cs
GEM/wiki_lingua_de
GEM/wiki_lingua_en
GEM/wiki_lingua_es
GEM/wiki_lingua_fr
GEM/wiki_lingua_hi
GEM/wiki_lingua_id
GEM/wiki_lingua_it
GEM/wiki_lingua_ja
GEM/wiki_lingua_ko
GEM/wiki_lingua_nl
GEM/wiki_lingua_pt
GEM/wiki_lingua_ru
GEM/wiki_lingua_th
GEM/wiki_lingua_tr
GEM/wiki_lingua_vi
GEM/wiki_lingua_zh
gem_xsum
gem_xsum_challenge_sample
gem_xsum_challenge_test_backtranslation
gem_xsum_challenge_test_bfp_02
gem_xsum_challenge_test_bfp_05
gem_xsum_challenge_test_covid
gem_xsum_challenge_test_nopunc
axb
axg
boolq
cb
cola
copa
crows_pairs_english
crows_pairs_french
diabla
e2e_nlg_cleaned
mnli
mnli_mismatched
multirc
piaf
qqp
rte
sst
tydiqa_primary
tydiqa_secondary
wic
wsc
wnli
wino_bias_type1_anti
wino_bias_type1_pro
wino_bias_type2_anti
wino_bias_type2_pro
xquad_ar
xquad_en
gsarti/flores_101_afr
gsarti/flores_101_amh
gsarti/flores_101_ara
gsarti/flores_101_hye
gsarti/flores_101_asm
gsarti/flores_101_ast
gsarti/flores_101_azj
gsarti/flores_101_bel
gsarti/flores_101_ben
gsarti/flores_101_bos
gsarti/flores_101_bul
gsarti/flores_101_mya
gsarti/flores_101_cat
gsarti/flores_101_ceb
gsarti/flores_101_zho_simpl
gsarti/flores_101_zho_trad
gsarti/flores_101_hrv
gsarti/flores_101_ces
gsarti/flores_101_dan
gsarti/flores_101_nld
gsarti/flores_101_eng
gsarti/flores_101_est
gsarti/flores_101_tgl
gsarti/flores_101_fin
gsarti/flores_101_fra
gsarti/flores_101_ful
gsarti/flores_101_glg
gsarti/flores_101_lug
gsarti/flores_101_kat
gsarti/flores_101_deu
gsarti/flores_101_ell
gsarti/flores_101_guj
gsarti/flores_101_hau
gsarti/flores_101_heb
gsarti/flores_101_hin
gsarti/flores_101_hun
gsarti/flores_101_isl
gsarti/flores_101_ibo
gsarti/flores_101_ind
gsarti/flores_101_gle
gsarti/flores_101_ita
gsarti/flores_101_jpn
gsarti/flores_101_jav
gsarti/flores_101_kea
gsarti/flores_101_kam
gsarti/flores_101_kan
gsarti/flores_101_kaz
gsarti/flores_101_khm
gsarti/flores_101_kor
gsarti/flores_101_kir
gsarti/flores_101_lao
gsarti/flores_101_lav
gsarti/flores_101_lin
gsarti/flores_101_lit
gsarti/flores_101_luo
gsarti/flores_101_ltz
gsarti/flores_101_mkd
gsarti/flores_101_msa
gsarti/flores_101_mal
gsarti/flores_101_mlt
gsarti/flores_101_mri
gsarti/flores_101_mar
gsarti/flores_101_mon
gsarti/flores_101_npi
gsarti/flores_101_nso
gsarti/flores_101_nob
gsarti/flores_101_nya
gsarti/flores_101_oci
gsarti/flores_101_ory
gsarti/flores_101_orm
gsarti/flores_101_pus
gsarti/flores_101_fas
gsarti/flores_101_pol
gsarti/flores_101_por
gsarti/flores_101_pan
gsarti/flores_101_ron
gsarti/flores_101_rus
gsarti/flores_101_srp
gsarti/flores_101_sna
gsarti/flores_101_snd
gsarti/flores_101_slk
gsarti/flores_101_slv
gsarti/flores_101_som
gsarti/flores_101_ckb
gsarti/flores_101_spa
gsarti/flores_101_swh
gsarti/flores_101_swe
gsarti/flores_101_tgk
gsarti/flores_101_tam
gsarti/flores_101_tel
gsarti/flores_101_tha
gsarti/flores_101_tur
gsarti/flores_101_ukr
gsarti/flores_101_umb
gsarti/flores_101_urd
gsarti/flores_101_uzb
gsarti/flores_101_vie
gsarti/flores_101_cym
gsarti/flores_101_wol
gsarti/flores_101_xho
gsarti/flores_101_yor
gsarti/flores_101_zul
)

#if [ "${#TASKS[@]}" -ne "$SLURM_ARRAY_TASK_COUNT" ];
#    then
#    echo "Please update the array size as the it doesn't correspond to the number of models we want to evaluate. Array size: $SLURM_ARRAY_TASK_COUNT, number of models: ${#TASKS[@]}"
#    exit 1
#fi
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

OPT_FOLDER=$WORK/opt/opt-175b/$TASK
mkdir -p $OPT_FOLDER

CMD="./tasks/eval_harness/evaluate_bsevalharness.py  \
    --load $CHECKPOINT_PATH \
    --results_path $OPT_FOLDER/bs_results.json \
    --tensor-model-parallel-size $TP_SIZE  \
    --pipeline-model-parallel-size $PP_SIZE \
    --micro-batch-size $EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --task_list $TASK\
    --deepspeed \
    --deepspeed_config ds_config.json \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier 16 \
    --offloadearly \
    $MEGATRON_REQUIRED_ARGS \
    "

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CUDA_LAUNCH_BLOCKING=1

echo $LAUNCHER $CMD

export PYTHONPATH=$MEGATRON_DEEPSPEED_REPO

$LAUNCHER $CMD 2>&1 | tee $OPT_FOLDER/bs-eval-harness.log
