# some debug notes / code - wip

Debug spikes suggestions:

- analyse restore from state_dict for word embedding

V try to comment out update lp params - with normal lr results are much worse - moved from 3.5 to 5.5 loss
- try again w/ lr=0

```
# deepspeed/runtime/bf16_optimizer.py
    def update_lp_params(self):
        return
```
- norm of activations to know which part gets out of sync
V run optimizer with LR=0
-

```
perl -le 'print qx[diff -u debug-emb-bf16-$_-pp0-tp0-dp0-$ARGV[1]-iteration.txt debug-emb-bf16-$_-pp1-tp0-dp0-$ARGV[1]-iteration.txt] for $ARGV[0]' 1 before


-



perl -le 'do { print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-save*.txt debug-$_-pp1-tp0-dp0-global1-on-save*.txt]; print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-load*.txt debug-$_-pp1-tp0-dp0-global1-on-load*.txt] } for 15..15'


perl -le 'do { print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-save*.txt debug-$_-pp0-tp0-dp0-global0-on-load*.txt]; print qx[diff -u debug-$_-pp1-tp0-dp0-global1-on-save*.txt debug-$_-pp1-tp0-dp0-global1-on-load*.txt] } for 15..15'



perl -le '$_=shift; do { print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-save*.txt debug-$_-pp1-tp0-dp0-global1-on-save*.txt]; print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-load*.txt debug-$_-pp1-tp0-dp0-global1-on-load*.txt] } ' 5


perl -le '$_=shift; do { print qx[diff -u debug-$_-pp0-tp0-dp0-global0-on-save*.txt debug-$_-pp0-tp0-dp0-global0-on-load*.txt]; print qx[diff -u debug-$_-pp1-tp0-dp0-global1-on-save*.txt debug-$_-pp1-tp0-dp0-global1-on-load*.txt] } ' 5


debug-12-pp0-tp0-dp0-global0-on-load-*.txt debug-13-pp0-tp0-dp0-global0-before-iteration-*.txt


debug-$_-pp0-tp0-dp0-global0-before-iteration-*.txt debug-$_-pp0-tp0-dp0-global0-after-iteration-*.txt




debug-bf16-21-pp0-tp1-before-iteration.txt
debug-bf16-21-pp0-tp1-after-iteration.txt


debug-fp32-21-pp1-tp0-before-iteration.txt
debug-fp32-21-pp1-tp0-after-iteration.txt




debug-bf16-21-pp0-tp1-before-iteration.txt
debug-bf16-21-pp0-tp1-after-iteration.txt


debug-fp32-21-pp1-tp0-before-iteration.txt
debug-fp32-21-pp1-tp0-after-iteration.txt










On each node - the same situation: (checked nodes 0 and 1 only)

6 processes are in:

Thread 835990 (active): "MainThread"
    train (megatron/training.py:915)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)

2 processes are in:

Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```

## debugging setup

```
salloc --partition=gpu_p5 --constraint=a100 --reservation=hug --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 --hint=nomultithread --gres=gpu:8 --time 20:00:00 --account=six@a100
```

```
bash 20B-n2-fp16.slurm
```

```
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile
```

```
ds_ssh -f hostfile "source ~/.pdshrc; ps aux | grep python | grep -v grep | grep `whoami` | awk '{print \$2}' | xargs -I {} py-spy dump --pid {} "
```

ps aux | grep python | egrep -v '(srun|grep)' | grep `whoami` | awk '{print $2}' | xargs -I {} py-spy dump --pid {}


must use `--gres=gpu:0` for the monitor or it'll block.
```
srun --gres=gpu:0 --jobid=$SLURM_JOBID ~/script.bash
```

```
srun --jobid=<jobid> --gres=gpu:0 -N <number_of_nodes> --tasks-per-node=1 --output=%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}'
```

```
#!/bin/bash

ps aux | grep python | grep -v grep | grep `whoami` | awk '{print $2}' | xargs -I {} py-spy dump --pid {}
```

```
ssh jean-zay-iam01 "~/script.bash"
```


# full debug

```
cd ~/prod/code/tr8b-104B/bigscience/train/tr11-200B-ml/

salloc --partition=gpu_p5 --constraint=a100 --reservation=hug --nodes=40 --ntasks-per-node=1 --cpus-per-task=64 --hint=nomultithread --gres=gpu:8 --time 20:00:00 --account=six@a100

bash 200B-n40-bf16-mono.slurm


# in another shell
squeue -u `whoami` -o "%.16i %.9P %.26j %.8T %.10M %.8l %.6D %.20S %R"
# adjust jobid
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

```
srun --jobid=4808 --gres=gpu:0 --nodes=48 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

# using pdsh

It's a bit tricky and doesn't work for `py-spy`


# tracing with python trace

This code will trace all python calls and log them to the console and into a dedicated per process log file.

This then can help to understand where some processes stopped responding, since we will have the log of the last call before it went unresponsive.



```
$ cat pretrain_gpt.py
[...]

def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

import re
class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __name__ == "__main__":

    import sys
    import trace
    import socket
    import os

    # enable to trace
    if 0:
        cwd = os.path.realpath('.')
        pid = os.getpid()
        hostname = socket.gethostname()
        local_rank = int(os.environ["LOCAL_RANK"])
        trace_output_file = f"{cwd}/trace-{hostname}-{local_rank}-{pid}.txt"

        # create a Trace object, telling it what to ignore, and whether to
        # do tracing or line-counting or both.
        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix],
            trace=1,
            count=1,
        )
        #    outfile=trace_output_file)

        # run the new command using the given tracer
        sys.stdout = Tee(trace_output_file)
        tracer.run('main()')
    else:
        main()

```
