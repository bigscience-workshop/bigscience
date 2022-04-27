# Data

* [Sharing data with the outside world](./export.md)



## Filesystem nuances

`WORK` has only 5M inodes, so we can't pre-process large datasets there. e.g. openwebtext is 8M files.

So need to pre-process it on `SCRATCH` and then copy over just the arrow files to `WORK` preserving the full pass as `datasets` expects it.

`SCRATCH` is also much faster as it's SSD, `WORK` is not!

After finishing pre-processing could tarball the raw files and put them to `STORE` for long term storage.

Remember, anything on `SCRATCH` gets wiped out after 30 days of not being accessed - or immediately at midnight of the same day if the file happens to have its access timestamp set in the past!

To clear out the empty dirs (ghosts of the once full dirs) run:

```
find /gpfswork/rech/six/commun/datasets/downloads/extracted -empty -type d -delete
find /gpfswork/rech/six/commun/datasets/ -empty -type d -delete
```


## Anatomy of `datasets` dataset:

Each stage is written in subdirectories of HF_DATASETS_CACHE so you can definitely rm any of these stages:

1. compressed source files are in `HF_DATASETS_CACHE / “downloads”`
2. uncompressed source files are in `HF_DATASETS_CACHE / “downloads” / “extracted”`
3. temporary build files are removed as soon as the dataset has been built (or if it failed) so in theory you shouldn’t have to do anything. But anyway they are at `HF_DATASETS_CACHE / <dataset_name> / <config_name> / <version> / <script_hash> + “.incomplete”`
4. cached arrow files are in `HF_DATASETS_CACHE / <dataset_name> / <config_name> / <version> / <script_hash>`

- So (1) can go to STORE
- 2 and 3 deleted
- 4 moved to WORK, preserving  `HF_DATASETS_CACHE / <dataset_name> / <config_name> / <version> / <script_hash>`

Of course, this will also require fiddling with `HF_DATASETS_CACHE` for the duration of this process to point to `WORK`.

## Code snippets

To get the full path expected by the cache checker:
```
from datasets.load import prepare_module, import_main_class
dataset_name = "openwebtext"
module_path, module_hash = prepare_module(dataset_name)
builder_cls = import_main_class(module_path)
builder = builder_cls(hash=module_hash)
print(builder.cache_dir)
#/Users/quentinlhoest/.cache/huggingface/datasets/openwebtext/plain_text/1.0.0/85b3ae7051d2d72e7c5fdf6dfb462603aaa26e9ed506202bf3a24d261c6c40a1
```

And ideally we want this:
```
from datasets import load_dataset_builder
dataset_name = "openwebtext"
dataset_builder = load_dataset_builder(dataset_name)
print(dataset_builder.cache_dir)
```
this feature was added in https://github.com/huggingface/datasets/pull/2500
