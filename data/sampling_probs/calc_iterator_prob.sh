mkdir -p $1
echo $1
echo $2
for alpha in .1 .2 .3 .4 .5 .6 .7 .8 .9; do
    python data/sampling_probs/calc_iterator_prob.py \
    --data-folder-path $2/ \
    --size-format GB \
    --alpha $alpha \
    --output-dir $1 \
    --name-prefix 'train' \
    --extension-name 'bin'
done
