export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/bigscience/tr3e-1B3-c4-checkpoints
cd tr3e-1B3-c4-checkpoints
$six_ALL_CCFRWORK/code/bigscience/tools/hub-sync.py --repo-path . --patterns '*bogus*'
git branch -a | sort -V | perl -lne 'm|(global_step\d+)| && print qx[git checkout $1; perl -pi -e "s|\\"tokenizer_class\\": null|\\"tokenizer_class\\": \\"T5Tokenizer\\"|" config.json; git commit -m "Fix tokenizer_class to use T5 tokenizer" .; git push --set-upstream origin $1]'
export GIT_LFS_SKIP_SMUDGE=0
