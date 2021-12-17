# tr11 200B ML

final size to be defined


## Size


existing models NLAYERS / NHIDDEN:
- 104B-ml: 64 / 11600
- gpt3 175B: 96 / 12288.
- meg 145B 80 / 12288
- meg 310B 96 / 16384

Possible ideas:

- 205B: 112 / 12288 (ratio: 109) narrow
- 206B: 96 / 13312 (ratio: 139) closer to normal 150-200 ratio

Formula to get model size, used 150k dict roughly - need to update:
```
NHIDDEN=12288;NLAYERS=112;SEQ_LEN=2048;VOCAB_SIZE=150257; python -c "h=$NHIDDEN; l=$NLAYERS; s=$SEQ_LEN; v=$VOCAB_SIZE; print(f'Model size: {(l*(12*h**2 + 13*h) + v*h + s*h + 2*h) / 10**9 :.0f}B')"
```
