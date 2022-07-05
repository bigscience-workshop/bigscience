# Checkpoint reshaping experiments

The reshaping experiments didn't do well. It seems that somehow changing PP impacted the loss - the higher the PP the worse the loss.

I think there are some subtle dependencies that we are missing here.


These are the results for resuming from `global_step94767_universal`


## 24 Nodes

TP=4/PP=12/MBS=2/Nodes=24

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 215.82 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.943883E+00 | grad norm: 0.128 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.490 | TFLOPs: 145.31 |
after restart
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 215.99 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.934365E+00 | grad norm: 0.133 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.482 | TFLOPs: 145.19 |
```

TP=4/PP=12/MBS=1/Nodes=24

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 234.72 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.943954E+00 | grad norm: 0.129 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 8.725 | TFLOPs: 133.61 |
```

TP=4/PP=24/MBS=1/Nodes=24

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 261.08 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.954172E+00 | grad norm: 0.133 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 7.844 | TFLOPs: 120.12 |
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 220.73 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.944833E+00 | grad norm: 0.132 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.279 | TFLOPs: 142.08 |
[default7]: iteration    94770/  115311 | consumed samples:    177931184 | consumed tokens: 364403064832 | elapsed time per iteration (s): 220.30 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.946273E+00 | grad norm: 0.136 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.296 | TFLOPs: 142.35 |
```

TP=2/PP=24/MBS=1/Nodes=24

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 221.87 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.945820E+00 | grad norm: 0.128 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.231 | TFLOPs: 141.34 |
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 199.82 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.936358E+00 | grad norm: 0.132 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 10.249 | TFLOPs: 156.95 |
```


## 36 Nodes


TP=4/PP=36/MBS=1/Nodes=36

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 219.17 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.963261E+00 | grad norm: 0.154 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 9.344 | TFLOPs: 95.39 |
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 155.71 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.953643E+00 | grad norm: 0.139 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 13.152 | TFLOPs: 134.26 |
```


TP=2/PP=36/MBS=1/Nodes=36

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 178.44 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 1.963201E+00 | grad norm: 0.153 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 11.477 | TFLOPs: 117.17 |
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 141.94 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.953437E+00 | grad norm: 0.149 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 14.428 | TFLOPs: 147.29 |
[default7]: iteration    94770/  115311 | consumed samples:    177931184 | consumed tokens: 364403064832 | elapsed time per iteration (s): 141.22 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 1.954596E+00 | grad norm: 0.154 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 14.502 | TFLOPs: 148.04 |
```

TP=1/PP=72/MBS=1/Nodes=36

```
[default7]: iteration    94768/  115311 | consumed samples:    177927088 | consumed tokens: 364394676224 | elapsed time per iteration (s): 168.97 | learning rate: 7.610E-06 | global batch size:  2048 | lm loss: 2.042900E+00 | grad norm: 1.335 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 12.120 | TFLOPs: 123.73 |
[default7]: iteration    94769/  115311 | consumed samples:    177929136 | consumed tokens: 364398870528 | elapsed time per iteration (s): 140.37 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 2.004192E+00 | grad norm: 0.511 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 14.590 | TFLOPs: 148.94 |
[default7]: iteration    94770/  115311 | consumed samples:    177931184 | consumed tokens: 364403064832 | elapsed time per iteration (s): 141.69 | learning rate: 7.609E-06 | global batch size:  2048 | lm loss: 2.000182E+00 | grad norm: 0.305 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 14.454 | TFLOPs: 147.55 |
```


matches:

TP=4/PP=36/MBS=1/Nodes=36
TP=2/PP=36/MBS=1/Nodes=36
