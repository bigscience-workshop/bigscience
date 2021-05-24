# Deepspeed notes


Q: on 8 gpus I get now: `data_parallel_size: 8, parameter_parallel_size: 8`

A: In this case seeing that the DP and parameter parallel size match means ZeRO will partition across all gpus

Q:
