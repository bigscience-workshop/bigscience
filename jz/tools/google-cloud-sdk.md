# google-cloud-sdk

Installed in `$six_ALL_CCFRWORK/lib/google-cloud-sdk`

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
