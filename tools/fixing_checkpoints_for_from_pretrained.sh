my_callback () {
  INDEX=${1}
  BRANCH=${2}
  if [[ $BRANCH == origin/global_step* ]];
  then
    git checkout "${BRANCH:7}"
    git mv "${BRANCH:7}"/* .
    cp ../gpt2_tokenizer/tokenizer.json .
    git add tokenizer.json
    git commit -m "fixed checkpoints to be from_pretrained-compatible"
    git push
  fi
}
get_branches () {
  git branch --all --format='%(refname:short)'
}
# mapfile -t -C my_callback -c 1 BRANCHES < <( get_branches ) # if you want the branches that were sent to mapfile in a new array as well
# echo "${BRANCHES[@]}"

export GIT_LFS_SKIP_SMUDGE=1
mapfile -t -C my_callback -c 1 < <( get_branches )
