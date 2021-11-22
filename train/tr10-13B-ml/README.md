# tr10 13B ML


## setup/tune up


To interactively tune up the setup:

```
salloc --constraint=v100-32g --account=six@gpu --nodes=4 --ntasks=4 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=120 bash --rcfile $six_ALL_CCFRWORK/code/tr10-13B/bigscience/train/tr10-13B-ml/start-tr10-13B
```


Conda setup:

```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n tr10-13B python=3.8
conda activate tr10-13B

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

mkdir $six_ALL_CCFRWORK/code/tr10-13B
cd $six_ALL_CCFRWORK/code/tr10-13B

cd $six_ALL_CCFRWORK/code/tr10-13B/apex
./build.sh

pip install deepspeed

cd $six_ALL_CCFRWORK/code/tr10-13B/DeepSpeed
./build.sh

pip install transformers

cd $six_ALL_CCFRWORK/code/tr10-13B/transformers
pip install -e .

cd $six_ALL_CCFRWORK/code/tr10-13B/megatron-lm
pip install -r requirements.txt
```

Env setup script to be `source start-tr10-13B` [start-tr10-13B](./start-tr10-13B)



configs:

works:
```
NNODES=4
TP_SIZE=4
PP_SIZE=4
```


tokenizer

It's at https://huggingface.co/teven/test_150k_vocab_tokenizer/tree/main !

So instead of running with :
```
--vocab-file $VOCAB_FILE \
--merge-file $MERGE_FILE \
```

You should run with:
```
--tokenizer-type PretrainedFromHF \
--tokenizer-name-or-path teven/test_150k_vocab_tokenizer \
```
￼￼
Preprocessed a c4 10k samples, you can use it with:
```
DATA_PATH=$six_ALL_CCFRSCRATCH/datasets-custom/150k_vocab_size_test/c4_10k_samples_150k_vocab_size
```

## Config


Julien Launay:

(1) the main difference will be multilinguality, and the larger vocabulary.
(2) For PrefixLM, we are not sure yet, as for now prefix is underperforming the vanilla model + it has some quirks. Thomas is working on a potential fix. We will keep you updated, but I think you can start working without prefix.
(3) Embeddings. ALiBi is still underperforming all others. Maybe we could consider going with rotary? @Iz Beltagy what's your opinion on this? Rotary probably won't change significantly your benchmark, but will degrade performance by a few percents across the board.
we don’t have a conclusive answer yet but both shouldn’t affect model size. If any, they will make the model a tiny bit smaller
(4) Activation. We need to evaluate the GeGLU run. GeGLU would bring a significant change to the size of the MLPs, which would be significant for your benchmark.
it shouldn’t change the overall model size but will change the size of some of the FF layers so might change how TP works

### `--init-method-std`

`--init-method-std 0.00884`

We derived this from: `NHIDDEN=5120`

`0.00884 = sqrt(2/(5120*5))` (from the ScaleNorm paper https://arxiv.org/abs/1910.05895)

### `NHEADS`

NHEADS=40, why...
