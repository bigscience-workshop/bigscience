# tr10 13B ML


## setup/tune up

```
srun --constraint=v100-32g --account=six@gpu  --pty --nodes=4 --ntasks=4 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=120 bash --rcfile $six_ALL_CCFRWORK/start-prod
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


`--init-method-std 0.00884`

We derived this from: `NHIDDEN=5120`

`0.00884 = sqrt(2/(5120*5))` (from the ScaleNorm paper https://arxiv.org/abs/1910.05895)
