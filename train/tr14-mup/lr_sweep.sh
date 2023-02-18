for lr in 0.052 0.017 0.0052 0.0017; do
    sbatch --job-name=tr14-39M-lr$lr-init0.1-inpm10-outm10-atnm10-mup tr14-39M-grid-search-mup.slurm $lr 0.1 10 10 10
done

for lr in 0.01 0.052 0.03 0.017 0.01 0.0052 0.003 0.0017 0.001; do
    sbatch --job-name=tr14-2B7-lr$lr-init0.1-inpm10-outm10-atnm10-mup tr14-2B7-grid-search-mup.slurm $lr 0.1 10 10 10
done
