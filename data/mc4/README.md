# mc4

## Megatron pre-processed files


These are the megatron-ready mc4 files:

XXX: todo:
- 1.3TB: `$six_ALL_CCFRWORK/datasets-custom/mc4/mc4_preprocessing`

Should something get corrupted there is a backup:

XXX: todo:
- 1.3TB: `$six_ALL_CCFRSTORE/datasets-custom/mc4/mc4_preprocessing`

If files need to re-pre-processed, the original jsonl files are at:

- 186GB: `$six_ALL_CCFRSTORE/datasets-custom/mc4/mc4_sampled_raw`


## How pre-processing was done

The pre-processing was done outside of JZ, and was downloaded from:

* [mc4_preprocessing](https://console.cloud.google.com/storage/browser/bigscience/mc4_preprocessing?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))
* [mc4_sampled_raw](https://console.cloud.google.com/storage/browser/bigscience/mc4_sampled_raw?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))

To download one needs to activate the already installed on JZ [google-cloud-sdk](../../jz/tools/google-cloud-sdk.md) and then use `gsutil` as instructed at the `Download` tab in the links above.
