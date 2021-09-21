# Train 6 - 104B - unmodified Megatron gpt2 - baseline - monolingual


## 104B Training

```
VOCAB_SIZE=50257 NLAYERS=32 NHIDDEN=16384 NHEADS=32 SEQ_LEN=2048; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l * (12*h**2 + 13*h) + (v * h) + (s * h) ) / 10**9 :.0f}B')"
104B
```

```
SEQLEN=2048

VOCAB_SIZE=50257
NLAYERS=32
NHIDDEN=16384
NHEADS=32
SEQ_LEN=2048
    --rampup-batch-size 16 16 6_000_000 \
    --global-batch-size 2048 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
```

- changed model shape/size
- double GBS
- changed lr and min-lr
