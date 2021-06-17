# Finetuning

Notes on the plans to do finetuning with the pre-trained model

# Large Model on smaller hardware setup

- fine-tuning a 150-200B model with fewer GPUs than the pre-training setup

## a. Fine-Tuning requiring only the model weights from the pre-training and uninitialized optimizer states


Solution: This can also be done using ZeRO-Infinity

Hardware Requirements: This would require about 2.5-5 TB of aggregate memory for 100-200B model. It can be either CPU memory or NVMe memory, and it can be within a single node or across nodes. A single node server with enough CPU or NVMe can work, if speed is not an issue.

Estimated Work: We can do this with ZeRO-Infinity. Seems like @Shaden Smith already has the code to load the model parameters checkpoints from Megatron+DeepSpeed 3D to Megatron+ DeepSpeed ZeRO-Infinity.

## b. Continued-Training requiring both the model weights and optimizer states after pre-training

Solution: This can be done using Megatron+DeepSpeed 3D with ZeRO CPU Offload.

Hardware Requirements: This option will require 2-4 TB of aggregate CPU memory to store the optimizer states and 600-1200GB of aggregate GPU memory to store parameters, gradients and activations for 100-200B model.

This reduces the number of GPUs required by 4x. Will run on 32-64 GPUs on 4-8x nodes with 8xV100, 768GB RAM.

Estimated work: The current code already supports it.
