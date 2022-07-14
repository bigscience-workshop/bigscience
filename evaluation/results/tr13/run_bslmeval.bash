"""
Setup:
Get bs/lm-eval-harness, install reqs + accelerate
"""
cd /content/lm-evaluation-harness
# For testing add:    --limit 20 \
python3 main.py --model hf-causal \
    --model_args pretrained=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/6b3global_step163750,use_accelerate=True,tokenizer=/gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3t0/6b3global_step163750,dtype=float16 \
    --tasks wnli \
    --device cuda \
    --no_cache \
    --num_fewshot 0
