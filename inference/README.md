# Inference

Notes on the plans to do inference with the pre-trained model

# Large Model on limited hardware

- inferencing and tinkering on a single host (150-200B model)

Solution: We can do this with ZeRO-Infinity. Seems like @Shaden Smith already has the code to load the model parameters checkpoints from Megatron+DeepSpeed 3D to Megatron+ DeepSpeed ZeRO-Infinity. The remaining work is to add an inference only mode to ZeRO-Infinity that drops all the non-parameter states.

Hardware Requirements : Would require about 500-1000 GB of memory (can be CPU, GPU or NVMe). Single Node with enough CPU or NVMe memory should work here.

Estimated Work: If all works as expected, 1-3 weeks based on bandwidth availability. Tuning for the best performance might another week or so, but that wont be blocking the availability of the functionality.
