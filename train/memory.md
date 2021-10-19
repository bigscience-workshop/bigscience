# Memory Utilization

# Activation Partitioning

> Activation Partitioning is a memory optimization in ZeRO that can reduce the memory consumed by activations during model parallel training (MP). In MP certain activations maybe required by all MP processes, resulting in a replication of activations across MP GPUs. Activation Partitioning stores these activations in a partitioned state once they are used for computation in the forward propagation. These activations are allgathered right before they are needed again during the backward propagation. By storing activations in a partitioned state, ZeRO in DeepSpeed can reduce the activation memory footprint proportional to the MP degree.

To activate add `--partition-activations`
