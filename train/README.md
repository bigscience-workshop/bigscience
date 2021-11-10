## Training scripts

This folder gathers training scripts for the different arch/scaling and engineering experiments. The naming convention is `tr<number>-<short-description>`. The current baseline that architecture and scaling experiments compare to is [tr3d](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr3-1B3-baseline/tr3d-1B3-more-warmup.slurm). In order to launch a new experiment, you should probably start from the [arch-and-scaling template](https://github.com/bigscience-workshop/bigscience/blob/master/train/arch-and-scaling-template.slurm).

Some tips:
 - [TFlops optimization](https://github.com/bigscience-workshop/bigscience/blob/master/train/tflops_optimization.md): How to make sure that given a set of hardware you optimize the speed at which you train.
 - [Instrumentation](https://github.com/bigscience-workshop/bigscience/blob/master/tools/README.md): How to sync with the hub
