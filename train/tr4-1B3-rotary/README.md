## Tr4c

On the supercomputer, we prepared the folders
```bash
cd $six_ALL_CCFRSCRATCH/synched_exps/
mkdir tr4c-1B3-rotary-oscar
huggingface-cli repo create tr4c-1B3-rotary-oscar-logs --organization bigscience
huggingface-cli repo create tr4c-1B3-rotary-oscar-checkpoints --organization bigscience
git clone tr4c-1B3-rotary-oscar-logs
git clone tr4c-1B3-rotary-oscar-checkpoints
mv tr4c-1B3-rotary-oscar-checkpoints checkpoints
cd tr4c-1B3-rotary-oscar-logs
mkdir logs
```

And then launch the jobs: 
```
sbatch --array=1-11%1 $SCRATCH/repos/bigscience/train/tr4-1B3-rotary/tr4c-1B3-oscar-modeling-rotary.slurm
```