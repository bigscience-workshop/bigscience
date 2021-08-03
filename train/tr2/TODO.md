# Train 2 TODO

- name the project (like tr2-26B-prompt)
    -  arch&scale suggests using the same model size as tr1 (13B) but with the model and data changes listed below

- group the tensorboard reports:

```
Batch-size
- Batch-size
- Batch-size vs samples
Grad-norm
- Grad norm
- Grad norm vs samples
Learning rate
- Learning rate
- Learning rate vs samples
Lm loss train
- Lm loss
- Lm loss vs samples
Lm loss validation
- Lm loss
- Lm loss vs samples
- Lm loss ppl
- Lm loss ppl vs samples
Loss scale
- Loss scale
- Loss scale vs samples
Num zeros
- Num zeros
- Num zeros vs samples
```
that's mostly about changing to

```
tb.add_scalar("batch size/batch size", batch_size, iteration)
tb.add_scalar("batch size/batch size vs samples", batch_size, args.consumed_train_samples)
```

tracking: https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/38

add new metrics: XXX

- Depending on the results from the arch&scale experiments (when do we expect to start this run? we want to make sure we have answers for the following questions by then)
    - Pre-layernorm (if it is not already used)
    - Rotary embeddings
    - Prefix-lm

- Train on multiple languages

