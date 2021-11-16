## Tr4c

On the supercomputer, we prepared the folders
```bash
cd $six_ALL_CCFRSCRATCH/synched_exps/
huggingface-cli repo create tr4c-1B3-rotary-oscar-logs --organization bigscience
huggingface-cli repo create tr4c-1B3-rotary-oscar-checkpoints --organization bigscience
mkdir tr4c-1B3-rotary-oscar
cd tr4c-1B3-rotary-oscar
git clone https://huggingface.co/bigscience/tr4c-1B3-rotary-oscar-logs
git clone https://huggingface.co/bigscience/tr4c-1B3-rotary-oscar-checkpoints
mv tr4c-1B3-rotary-oscar-checkpoints checkpoints
cd tr4c-1B3-rotary-oscar-logs
mkdir logs
```

And then launch the jobs: 
```
sbatch --array=1-11%1 $SCRATCH/repos/bigscience/train/tr4-1B3-rotary/tr4c-1B3-oscar-modeling-rotary.slurm
```