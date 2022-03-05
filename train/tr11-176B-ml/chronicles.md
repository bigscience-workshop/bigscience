# The final training

Trials and Tribulations during the 176B 250k multi-lingual training.

## main-1

While the final data is being cleaned up we are doing a few preliminary runs with data that still has some issues.

GBS ramp up of `--rampup-batch-size 16 16 9_765_625` - the first few stages starting with GBS=16 are really slow (8 TFLOPs). The pipeline doesn't have enough data to even fill all the stages once, so it's super inefficient and it'll take days until we start hitting 100 TFLOPs.

But there were no spikes during this brief experiment.



## main-2

Trying `--rampup-batch-size 384 16 9_765_625` since 384 is the first GBS where the pipe is filled up fully for the first time. `12*2*4=384` (`PP*MBS*DP`). The throughput start at 100 TFLOPs right away (and it should be 150 TFLOPS once we reach GBS=2048).

Found a bug: tied weights weren't getting reduced - was getting a spike on restart, fixed at
https://github.com/microsoft/DeepSpeed/pull/1801/commits/37011a92bad42b07c2cb742751873ef7073d84b8

So only the front embed matrix grad updates were making, the end one were ignored.

Will do a totally new run to compare that it's similar or better.


## main-3
