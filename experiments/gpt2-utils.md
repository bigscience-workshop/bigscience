# Handy info utils for gpt2


## Calculate model size

Calculate the number of params in the model:

- h = hidden size
- l = num_layers
- s = sequence length
- v = vocabulary size

```
$ python -c 'h=1024; l=24; s=1024; v=50257; print(f"{l * (12*h**2 + 13*h) + (v * h) + (s * h) >> 20}M")'
338M
```

Full math for the above final formula: (num_heads is not really part of the calculations)

```# Let h = hidden size, n = num_layers, k = num_heads, s = sequence length, v = vocabulary size
# Compute Embedding Parameters (Vocab + Position)
emb_params = (v * h) + (s * h)
# Compute Parameters per Transformer Block
head_dim = h / k
qkv_params_w = k * (3 * (h * (h / k))) = 3 * h * h    # 3h^2
mh_reduce_w = (k * ((h / k)) * h = h * h              #  h^2
qkv_params_b = k * (3 * (h / k)) = 3 * h              # 3h
mh_reduce_b = h                                       #  h
pos_ff_exp_w = h * (4 * h)                            # 4h^2
pos_ff_con_w = (4 * h) * h                            # 4h^2
pos_ff_exp_b = 4 * h                                  # 4h
pos_ff_con_b = h                                      #  h
layer_norm1 = 2 * h                                   # 2h
layer_norm2 = 2 * h                                   # 2h
# Magic Formula:
total_params = n * (12h^2 + 13h) + (v * h) + (s * h)

```


Can calculate the same on a given `model` object (counts shared params once):
```
sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
```
