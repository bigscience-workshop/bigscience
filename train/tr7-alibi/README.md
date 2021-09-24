## Tr7a

On the supercomputer, we prepared the folders
```bash
cd $six_ALL_CCFRSCRATCH/synched_exps/
mkdir tr7a-1B3-alibi
huggingface-cli repo create tr7a-1B3-alibi-logs --organization bigscience
huggingface-cli repo create tr7a-1B3-alibi-checkpoints --organization bigscience
git clone tr7a-1B3-alibi-logs
git clone tr7a-1B3-alibi-checkpoints
mv tr7a-1B3-alibi-checkpoints checkpoints
cd tr7a-1B3-alibi-logs
mkdir logs
```

And then launch the jobs: 
```
sbatch --array=1-11%1 $SCRATCH/repos/bigscience/train/tr7-alibi/tr7a-1B3-modeling-alibi.slurm
```

To sync tensorboard:
```
$SCRATCH/repos/bigscience/tools/hub-sync.py --repo-path  $ALL_CCFRSCRATCH/synched_exps/tr7a-1B3-alibi/tr7a-1B3-alibi-logs/ --patterns '*tfevents*' -d
```