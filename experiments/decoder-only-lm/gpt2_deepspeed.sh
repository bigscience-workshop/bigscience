DATASET=openwebtext
SERIALIZATION_DIR=${ALL_CCFRWORK}/experiments/gpt2_repro
LOGGING_DIR=${ALL_CCFRWORK}/tensorboard/gpt2_repro

export CUDA_VISIBLE_DEVICES=0

#python scripts/run_clm.py \
deepspeed scripts/run_clm.py \
    --deepspeed configs/deepspeed/ds_zero2.json \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name openwebtext --block_size 1024 \
    --preprocessing_num_workers 76 \
    --group_by_length --length_column_name length \
    --do_train --do_eval \
    --max_steps 15000 \
    --max_train_samples 10000000 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 8 \
    --output_dir outputs --overwrite_output_dir \
    --report_to tensorboard \
    --logging_strategy steps --logging_first_step --logging_dir logs --logging_steps 20 \
    --eval_steps 250 --evaluation_strategy steps \
    --save_strategy steps --save_steps 500 --save_total_limit 31 \
    --n_layer 3 --n_embd 128 --n_inner 128 --n_head 8
