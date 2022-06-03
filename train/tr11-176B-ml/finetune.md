# Finetuning 176B

Finetuning 176B at the end of the training might be necessary to ensure exact logits match between Megatron-DS trained model and HF model.
For now, there are 2 main bottlenecks that are responsible of not giving 100% logits match between HF model and Megatron model

## Diverging bottlenecks

### TP merging strategy

See [this issue](https://github.com/pytorch/pytorch/issues/76232). When merging TP ranks the logits exactness is lost. The idea would be to finetune the 176B model with TP=1

### Use `torch_softmax` instead of `fused_softmax`

`fused_softmax` and `torch_softmax` does not give the same results (ie, `torch.testing.assert_allclose(atol=0.0, rtol=0.0)` does not pass). The main model could be finetuned with `torch_softmax`.
See [this line](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/cb48bd2c8bc182fb9872f127ef7c2267fbf9cc2e/megatron/model/fused_softmax.py#L204)
