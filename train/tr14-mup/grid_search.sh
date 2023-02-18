for inpm in 10 1 0.1 0.01 0.001
do
    for outm in 10 1 0.1 0.01 0.001
    do
        for atnm in 10 1 0.1 0.01 0.001
        do
            for lr in 0.1 0.03 0.01 0.003 0.001
            do
                for init in 1 0.3 0.1 0.03 0.01
                do
                    sbatch --job-name=tr14-39M-lr$lr-init$init-inpm$inpm-outm$outm-atnm$atnm-mup tr14-39M-grid-search-mup.slurm $lr $init $inpm $outm $atnm
                done
            done
        done
    done
done
