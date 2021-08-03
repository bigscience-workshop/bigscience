# Train 2 TODO

- name the project (like tr2-26B-prompt)

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
tb.add_scalar("batch size/batch size", batch_size, iteration)
and tb.add_scalar("batch size/batch size vs samples", batch_size, args.consumed_train_samples)

add new metrics: XXX
