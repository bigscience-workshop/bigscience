# Backup Schedule

First start an internet instance that won't get killed:

```
srun --pty -A six@cpu -p compil --hint=nomultithread --time=20:00:00 bash --rcfile $six_ALL_CCFRWORK/start-prod
```

then back up:

## logs and eval-results (tiny)

```
gsutil rsync -x ".git" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/tr11-176B-ml-logs  gs://bigscience-backups/tr11-176B-ml/tr11-176B-ml-logs
gsutil rsync -x ".git" -r $six_ALL_CCFRSTORE/checkpoints/tr11-176B-ml/eval-results gs://bigscience-backups/tr11-176B-ml/tr11-176B-ml-eval-results
```


## full checkpoint (2.3TB)

12 checkpoints: total 27TB

```
# done

gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step3000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step3000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step10000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step10000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step20000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step20000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step30000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step30000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step40000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step40000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step50000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step50000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step60000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step60000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step70000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step70000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step80000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step80000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step90000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step90000

# in-progress



# todo:

gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step100000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step100000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step110000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step110000
gsutil rsync -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step120000 gs://bigscience-backups/tr11-176B-ml/checkpoints/global_step120000

```



## weights only checkpoints (0.33TB)


40 checkpoints: total 13TB - autogenerate the schedule:
```
perl -le 'print qq[gsutil rsync -x "bf16.*" -r \$six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step$_ gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step$_] for map { $_*3000 } 1..40'
```

```
# done

gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step3000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step3000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step6000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step6000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step9000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step9000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step12000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step12000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step15000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step15000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step18000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step18000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step21000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step21000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step24000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step24000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step27000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step27000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step33000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step33000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step36000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step36000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step39000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step39000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step42000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step42000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step45000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step45000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step48000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step48000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step51000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step51000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step54000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step54000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step57000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step57000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step63000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step63000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step66000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step66000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step69000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step69000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step72000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step72000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step75000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step75000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step78000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step78000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step81000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step81000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step84000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step84000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step87000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step87000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step93000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step93000

# in-progress



# todo:

gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step96000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step96000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step99000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step99000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step102000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step102000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step105000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step105000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step108000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step108000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step111000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step111000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step114000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step114000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step117000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step117000
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSCRATCH/checkpoints/tr11-176B-ml/checkpoints/main/global_step120000 gs://bigscience-backups/tr11-176B-ml/checkpoints-weights/global_step120000

```



## spikes

weights only

```
# done



gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSTORE/checkpoints/tr11-176B-ml/checkpoints/spikes/global_step31200 gs://bigscience-backups/tr11-176B-ml/checkpoints-spikes-weights/global_step31200
gsutil rsync -x "bf16.*" -r $six_ALL_CCFRSTORE/checkpoints/tr11-176B-ml/checkpoints/spikes/global_step31259 gs://bigscience-backups/tr11-176B-ml/checkpoints-spikes-weights/global_step31259

# in-progress



# todo:

```


## Tarring the checkpoints in STORE

Since we don't have too many inodes in STORE we ought to tar the checkpoints

```
cd /gpfsdsstore/projects/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints
cd 1
find * -maxdepth 0 -type d -exec tar cvf {}.tar {} \;

```
