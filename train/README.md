## Training scripts

This folder gathers training scripts for the different arch/scaling and engineering experiments. The naming convention is `tr<number>-<short-description>`. The current baseline that architecture and scaling experiments compare to is [tr3d](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr3-1B3-baseline/tr3d-1B3-more-warmup.slurm). In order to launch a new experiment, you should probably start from the [arch-and-scaling template](https://github.com/bigscience-workshop/bigscience/blob/master/train/arch-and-scaling-template.slurm).

Some tips:
 - [TFlops optimization](https://github.com/bigscience-workshop/bigscience/blob/master/train/tflops_optimization.md): How to make sure that given a set of hardware you optimize the speed at which you train.
 - [Instrumentation](https://github.com/bigscience-workshop/bigscience/blob/master/tools/README.md): How to sync with the hub

## Stored checkpoints

Location of the checkpoints of the trained models plus logs and anything else of importance - e.g. eval harness results:

- tr1-13B: `gs://bigscience-backups/tr1-13B/`

- tr3m-1B3-emb-norm-pile: `$six_ALL_CCFRSTORE/checkpoints/tr3m-1B3-emb-norm-pile`

- tr4-1B3-rotary: `$six_ALL_CCFRSTORE/checkpoints/
- tr4b-350M-rotary: `$six_ALL_CCFRSTORE/checkpoints/
- tr4c-1B3-rotary-oscar: `$six_ALL_CCFRSTORE/checkpoints/tr4c-1B3-rotary-oscar`

- tr6-1B3-prefix-lm: `$six_ALL_CCFRSTORE/checkpoints/tr6-1B3-prefix-lm`
- tr6-1B3-prefix-lm-unbiased-loss: `$six_ALL_CCFRSTORE/checkpoints/tr6-1B3-prefix-lm-unbiased-loss`
- tr6b-350M-prefix-lm: `$six_ALL_CCFRSTORE/checkpoints/tr6b-350M-prefix-lm`
- tr6b-350M-prefix-lm-PP2: `$six_ALL_CCFRSTORE/checkpoints/tr6b-350M-prefix-lm-PP2`
- tr6b-350M-prefix-lm-unbiased-loss: `$six_ALL_CCFRSTORE/checkpoints/tr6b-350M-prefix-lm-unbiased-loss`
- tr6c-350M-prefix-lm-reset-attention-mask: `$six_ALL_CCFRSTORE/checkpoints/tr6c-350M-prefix-lm-reset-attention-mask`
- tr6c-350M-prefix-lm-reset-attention-mask.backup: `$six_ALL_CCFRSTORE/checkpoints/tr6c-350M-prefix-lm-reset-attention-mask.backup`
- tr6d-350M-prefix-lm-pile: `$six_ALL_CCFRSTORE/checkpoints/tr6d-350M-prefix-lm-pile`
- tr6e-1B3-pile: `$six_ALL_CCFRSTORE/checkpoints/tr6e-1B3-pile`
- tr6f-1B3-oscar-no-loss-on-targets-only: `$six_ALL_CCFRSTORE/checkpoints/tr6f-1B3-oscar-no-loss-on-targets-only`
- tr6g-1B3-oscar-loss-reweighting: `$six_ALL_CCFRSTORE/checkpoints/tr6g-1B3-oscar-loss-reweighting`

- tr7a-1B3-alibi (not a real alibi pos embedding experiment - the alibi matrix were not used in this experiment): `$six_ALL_CCFRSTORE/checkpoints/tr7a-1B3-alibi`
- tr7b-350-alibi (not a real alibi pos embedding experiment - the alibi matrix were not used in this experiment): `$six_ALL_CCFRSTORE/checkpoints/tr7b-350M-alibi`
- tr7d-1B3-alibi: `six_ALL_CCFRSTORE/checkpoints/tr7d-1B3-alibi`

- tr9b-350M-swiglu: `six_ALL_CCFRSTORE/checkpoints/tr9b-350M-swiglu`
- tr9c-1B3-swiglu-pile: `six_ALL_CCFRSTORE/checkpoints/tr9b-1B3-swiglu-pile`

- tr13: Multi-Task Fine-tuning (T0)
