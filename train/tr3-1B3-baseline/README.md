# Arch/Scaling baselines (tr3)

This folder contains the training scripts for the architecture and scaling baseline runs: no fancy tricks, just GPT2. Here are links to the respective tensorboards:

| Size                	| 1B3 	| 760M 	| 350M 	| 125M 	|
|---------------------	|-----	|------	|------	|------	|
| C4 + low warmup     	| [a](https://huggingface.co/bigscience/tr3-1B3-modeling-baseline-tensorboard)   	| [b](https://huggingface.co/bigscience/tr3b-760M-modeling-baseline-tensorboard)    	| [c](https://huggingface.co/bigscience/tr3c-350M-modeling-baseline-tensorboard)    	|      	|
| OSCAR + low warmup  	| [f](https://huggingface.co/bigscience/tr3f-1B3-diagnostic2-low-warmup-oscar-tensorboard)   	|      	|      	|      	|
| C4 + high warmup    	| [e](https://huggingface.co/bigscience/tr3e-1B3-diagnostic1-warmup-c4-tensorboard)   	|      	|      	|      	|
| OSCAR + high warmup 	| **[d (current baseline)](https://huggingface.co/bigscience/tr3d-1B3-more-warmup-tensorboard)**   	| [g](https://huggingface.co/bigscience/tr3g-760M-v2-tensorboard)    	| [h](https://huggingface.co/bigscience/tr3h-350M-v2-tensorboard)    	| [i](https://huggingface.co/bigscience/tr3i-125M-v2-tensorboard)    	|
| Pile + high warmup  	| [m](https://huggingface.co/bigscience/tr3m-1B3-pile-tensorboard)   	| [j](https://huggingface.co/bigscience/tr3j-760M-pile-tensorboard)    	| [k](https://huggingface.co/bigscience/tr3k-350M-pile-tensorboard)    	| [l](https://huggingface.co/bigscience/tr3l-125M-pile-tensorboard)    	|



# emb-norm

a full re-run of `tr3m-1B3-pile-tensorboard` with `--embed-layernorm` enabled

[script](tr3m-1B3-emb-norm-pile.slurm)

results:

- added `emb-norm` to https://huggingface.co/bigscience/tr3m-1B3-pile-tensorboard/tensorboard and moved the original run to `base`. Also upgraded the old TB to the new format so the new graph names match.

- full standalone repo: https://huggingface.co/bigscience/tr3m-1B3-emb-norm-pile-logs/ with logs

- last checkpoint saved in `$six_ALL_CCFRSTORE/checkpoints/tr3m-1B3-emb-norm-pile`
