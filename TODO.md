# Things to do

## Carbon Footprint Tracking

Instrument multi-node carbon footprint tracking. https://github.com/mlco2/codecarbon
Seems like we only need to add about 2 lines in our code.

The decision is to run it on one node (gpu?), and then the results will be multiplied by the number of nodes. It generates a csv results. Need to figure out where to broadcast it to from JZ.

Blocking event: no codebase to yet to add it to - need to fork https://github.com/microsoft/Megatron-DeepSpeed once it's ready.


## Weights-Only checkpoints

Contributors that have no access to JZ will want to have intermediary checkpoints to work with. It'll be very slow to scp full checkpoints. Would it be possible to either post-process the Deepspeed PP checkpoints and extract just the model weights before copying those from JZ?

The current DS PP format saves each layer's state dict in its own file, and they're named differently than the optimizer states. Could be as simple as pattern matching the scp. The pipeline engine selectively loads the files based on pipeline rank, so no need to merge them.

But users outside of JZ will very likely have a different HW setup, so these will need to be re-shaped to match a new PP-degree.
