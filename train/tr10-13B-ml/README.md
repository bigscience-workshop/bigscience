# tr10 13B ML


## setup/tune up

```
srun --constraint=v100-32g --account=six@gpu  --pty --nodes=4 --ntasks=4 --cpus-per-task=40 --gres=gpu:4 --hint=nomultithread --time=120 bash --rcfile $six_ALL_CCFRWORK/start-prod
```


```
export CONDA_ENVS_PATH=$six_ALL_CCFRWORK/conda

conda create -y -n tr10-13b python=3.8
conda activate tr10-13b

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install deepspeed

mkdir $six_ALL_CCFRWORK/code/tr10-13b

cd $six_ALL_CCFRWORK/code/transformers
pip install -e .[dev]

cd $six_ALL_CCFRWORK/code/megatron-lm
pip install -r requirements.txt

cd $six_ALL_CCFRWORK/code/apex
./build.sh

cd $six_ALL_CCFRWORK/code/deepspeed
./build.sh

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
I preprocessed a c4 10k samples, you can use it with

DATA_PATH=/gpfsscratch/rech/six/commun/datasets-custom/150k_vocab_size_test/c4_10k_samples_150k_vocab_size
￼￼
