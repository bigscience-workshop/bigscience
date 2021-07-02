# Things to do

## Carbon Footprint Tracking

Instrument multi-node carbon footprint tracking. https://github.com/mlco2/codecarbon
Seems like we only need to add about 2 lines in our code.

The decision is to run it on one node (gpu?), and then the results will be multiplied by the number of nodes. It generates a csv results. Need to figure out where to broadcast it to from JZ.

Blocking event: no codebase to yet to add it to - need to fork https://github.com/microsoft/Megatron-DeepSpeed once it's ready.
