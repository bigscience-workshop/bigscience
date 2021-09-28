# tr8-104B Chronicles

Notes on the training progress with a particular focus on any encountered problems and their diagnosis and solutions/prevention.

To follow the training progress charts, see:  [tensorboard](https://huggingface.co/bigscience/tr8-104B-logs/tensorboard)

To follow the raw training logs see: [logs](https://huggingface.co/bigscience/tr8-104B-logs/tree/main/logs)

# Glitch 1

- Nodes: `64`
- Seed: `42`
- Started from iteration 0

![tr8-104B-glitch-1.png](images/tr8-104B-glitch-1.png)

Somewhere between iteration 7000 and 7010 lm loss jumped from 6.4 to 14 and then 200 iterations later it went down to ~7 and stayed there w/o any change, and later it went into NaN.

```
 iteration     7000/  159576 | consumed samples:       260912 | elapsed time per iteration (ms): 18706.1 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 6.444662E+00 | loss scale: 2048.0 | grad norm: 98258.265 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 7000 | lm loss value: 7.174200E+00 | lm loss PPL: 1.305315E+03 |
------------------------------------------------------------------------------------------------
 iteration     7010/  159576 | consumed samples:       261872 | elapsed time per iteration (ms): 19904.0 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 1.142026E+01 | loss scale: 2048.0 | grad norm: 219645.978 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [...]
 iteration     7220/  159576 | consumed samples:       282032 | elapsed time per iteration (ms): 18333.4 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 7.155109E+00 | loss scale: 2048.0 | grad norm: 16921.991 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |

```


Training more hasn't helped at all

Solution:
- roll back to the last good checkpoint `global_step6210`
- change seed. New seed `43`.

Rollback:
1. moved all checkpoints after `global_step6210` to another dir
2. couldn't leave tensorboard files from the unrolled section as is, so fixed tensorboard by first copying all the existing events log files to a new dir
```
cd /gpfsscratch/rech/six/commun/checkpoints/tr8-104B/tr8-104B-logs/tensorboard
mkdir tb-7k-glitch
cp events* tb-7k-glitch
git add tb-7k-glitch
git commit -am "saved the original tensorboard logs"
git push
```
now checking the timestamp of the last checkpoint `global_step6210` we are rolling from and now manually removing all event log files from the main log whose timestamp is newer than the checkpoint `global_step6210`

now we have 2 tensorboards - the main running one and the one which we couldn't recover from - but we want it for posterity

Having a new seed forced regeneration of `.pny` files which re-randomized the order. If the glitch were due to faulty data this should have fixed the problem.

Started a new training from the last good checkpoint and it run until we run into a similar glitch even sooner.

# Glitch 2


- Nodes: `64`
- Seed: `43` (from the beginning)
- Restarted from `global_step6210`

![tr8-104B-glitch-1.png](images/tr8-104B-glitch-1.png)


Similar to glitch 1, but even sooner we went from 6.3 to 9 to 6.7


```
 iteration     6900/  159576 | consumed samples:       251312 | elapsed time per iteration (ms): 18495.6 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 6.365808E+00 | loss scale: 4096.0 | grad norm: 95313.572 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6910/  159576 | consumed samples:       252272 | elapsed time per iteration (ms): 18802.1 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 6.598378E+00 | loss scale: 4096.0 | grad norm: 84678.880 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6920/  159576 | consumed samples:       253232 | elapsed time per iteration (ms): 18641.0 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 7.314456E+00 | loss scale: 4096.0 | grad norm: 122716.232 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6930/  159576 | consumed samples:       254192 | elapsed time per iteration (ms): 18564.1 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 9.121927E+00 | loss scale: 4096.0 | grad norm: 283384.130 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration     6940/  159576 | consumed samples:       255152 | elapsed time per iteration (ms): 18549.7 | learning rate: 6.000E-05 | global batch size:    96 | lm loss: 1.023865E+01 | loss scale: 4096.0 | grad norm: 42359.376 | num zeros: 0.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```

Conglong Li made an interesting observation that in both cases the glitch happened very closely to the moment where LR warmup stopped, to quote:

> The gradients are the largest at the end of the LR ramp-up phase, so that's when there the training is the most unstable. There is no easy fix. Curriculum learning helps, and so does potentially lengthening warm-up/reducing the LR.
>
> We logged the L1 norm/max element of Adam optimizer's gradient variance, and found that 1) the norm and max element has a correlation with the LR schedule: they all reach max at/near LR peak 2) that is also where we have the highest risk of divergence.

Moreover, we reviewed the tr1-13B training and we had a huge glitch there from which it recovered, perhaps since the model was much smaller.

![tr1-13B-glitch-1-2.png](images/tr1-13B-glitch-1-2.png)

There the LR rampup stopped around 25k, and the first huge glitch occurred at around 29k iteration
https://huggingface.co/bigscience/tr1-13B-tensorboard/tensorboard
According to Conglong Li 25k and 29k are close enough based to their study. Quoting him:

> In our study of [1.5B gpt-2](https://arxiv.org/pdf/2108.06084.pdf), we used 3K LR warmup and here you can see the grad variance norm (left) only reach bottom at 8K+ steps, and baseline's grad var max is unstable during first 10K+ steps:

![step-wise-adam-variance](images/step-wise-adam-variance-1.png)

So it looks that now we have 3 documented glitches that all are related to the end of the LR warm up end.


We are not yet excluding the case that something is wrong with the data. Going to look into it next.



XXX: to be continued

stopped at Date: 2021-09-27
