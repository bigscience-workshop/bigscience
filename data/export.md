# Export Data outside of JZ



## Upload to the Hub

First go to https://huggingface.co/bigscience/ and via your username (right upper corner) create "new Model"
while choosing the `bigscience` as org.

Say you created https://huggingface.co/bigscience/misc-test-data/

Now on JZ side

```
module load git-lfs
git lfs install
git clone https://huggingface.co/bigscience/misc-test-data/
cd misc-test-data/
```

Now you can add files which are less than 10M, commit and push.

Make sure that if the file is larger than 10M its extension is tracked by git LFS, e.g. if you're adding `foo.tar.gz` make sure `*gz` is in `.gitattributes` like so:
```
*.gz filter=lfs diff=lfs merge=lfs -text
```
if it isn't add it:
```
git lfs track "*.gz"
git commit -m "compressed files" .gitattributes
git push
```
only now add your large file `foo.tar.gz`
```
cp /some/place/foo.tar.gz .
git add foo.tar.gz
git commit -m "foo.tar.gz" foo.tar.gz
git push
```

Now you can tell the contributor on the other side where they can download the files you have just uploaded by sending them to the corresponding hub repo.
