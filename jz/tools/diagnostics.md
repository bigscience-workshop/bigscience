# Tools for diagnostics of training problems


## Hanging processes


To track down the culprit of a hung process dumping the stack traces of the training processes.
```
pgrep -f pretrain_gpt | xargs -i /path/to/py-spy dump --pid {} > /networked/path/unique/for/node
```

Given the dumps of a hung 3D trainer, the node with issues usually get stuck in a different part of the training pipeline. Pipelines with no issues will be waiting at an all-reduce before step, whereas the problematic pipeline usually hangs somewhere in the training microbatches. We often see the pipeline-adjacent processes stuck on a pipe send/recv from the problematic node(s).

If `py-spy` isn't already installed, do:
```
pip install py-spy
```


## Malfunctioning GPUs

Usually these require a reboot as once a problem happens on a hardware level, the recovery is not possible w/o a reboot.

For example if a GPU can't allocate memory because it has a hardware issue, as simple test could be:

```
python -c "import torch; torch.ones(1).cuda()"
```
