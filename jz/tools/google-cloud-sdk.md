# google-cloud-sdk

Installed in `$six_ALL_CCFRWORK/lib/google-cloud-sdk` following the linux installation instructions [here](https://cloud.google.com/sdk/docs/install?hl=en).

To activate add to your `~/.bashrc`:

```
if [ -f '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/path.bash.inc' ]; then . '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/path.bash.inc'; fi
if [ -f '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/completion.bash.inc' ]; then . '/gpfsssd/worksf/projects/rech/six/commun/lib/google-cloud-sdk/completion.bash.inc'; fi

```

and restart `bash`.

# Downloading from the `bigscience` bucket

Go to the location to download, e.g.:
`https://console.cloud.google.com/storage/browser/bigscience/mc4_preprocessing?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))`

Select dirs to download and click on 'Download` and it will give instructions to download all the dirs using `gsutil`, e.g.:

```
gsutil -m cp -r \
  "gs://bigscience/mc4_sampled_raw/am/" \
  "gs://bigscience/mc4_sampled_raw/ar/" \
  .
```

To debug add `-d`.

To download a single file, go to the file's page, e.g.:

https://console.cloud.google.com/storage/browser/_details/bigscience/mc4_preprocessing/en/train_text_document_1.bin

and it'll have the `gsutil URI` entry, in this case:  `gs://bigscience/mc4_preprocessing/en/train_text_document_1.bin` which you then feed to `gsutil`:

```
gsutil -m cp "gs://bigscience/mc4_preprocessing/en/train_text_document_1.bin" .
```

rsync might be a better way to sync files when they are large and the client keeps on crashing, example:
```
gsutil -m rsync -r "gs://bigscience/mc4_preprocessing" mc4_preprocessing
```
note that `gsutil` keeps track of what it failed to do and tries to re-do it even if you manually fetched a large file and inserted it into the right location, it'll ignore its appearance, will delete it and will attempt to fetch it a new. Not really great `rsync` feature, if you're used to the normal `rsync(1)` tool.

## moving multiple folders


`gsutil mv` is supposed to support globbing, but it doesn't. so here is a poor man's workaround:

e.g. to move `"gs://bigscience-backups/tr1-13B/global_step*"` to  `"gs://bigscience-backups/tr1-13B/checkpoints-bak/"`

```
for x in `gsutil ls "gs://bigscience-backups/tr1-13B"`; do y=$(basename -- "$x");echo gsutil mv ${x} gs://bigscience-backups/tr1-13B/checkpoints-bak/${y}; done > cmd
```
edit `cmd` to your liking to remove any folders that shouldn't be moved. surely can be further improved to filter out the wanted pattern, but the principle is clear.
