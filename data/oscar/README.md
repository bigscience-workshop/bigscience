# OSCAR

## Pre-processing

1. Convert `datasets` to `jsonl` which is the format required by Megatron-LM

The main script is [oscar-to-jsonl.py](./oscar-to-jsonl.py). Edit to change languages to use, initially using just English.

Note, that since shuffling slows the writeout process by 5-7 times, we don't shuffle in the script, but post-process it externally. See step 3.

To launch: [oscar-to-jsonl.slurm](./oscar-to-jsonl.slurm).

With "unshuffled_deduplicated_en" after filtering large entries (`>=1024`) we end up with 70754K examples out of 304230K total (about 1/4th of the full dataset).

The result is 5 files `oscar-[0-4].jsonl` of about 180GB each.

Runtime: ??h to download, ~2h to build, ~8h to filter, ~1.5h to write shards out


2. Concatenate

```
cat oscar-[0-4].jsonl > oscar.jsonl
```

This gives us a 900GB file.

Check:
```
$ wc -l oscar.jsonl
70754078 oscar.jsonl
```

Runtime: a few minutes



3. Shuffle

Megatron requires users to do their own shuffling of jsonl input.

It was too slow to do inside the filtering script, so we are using a post-processing solution.
Using https://github.com/alexandres/terashuf and 150GB RAM in ~1.5h we shuffle the file.

Important: note that the slurm job uses SCRATCH for `TMPDIR` and also sets the memory limit it can use to 150.0 (GB) (slightly under 160GB available on this slurm allocation to allow for other processes).

To launch: [oscar-fast-shuffle.slurm](./oscar-fast-shuffle.slurm)

`terashuf` is in `$six_ALL_CCFRWORK/bin/terashuf`

The result is `oscar-shuffled.jsonl`

Runtime: 1.5h



4. Megatron-LM preprocess

Finally we do the pre-processing:

To launch: [oscar-jsonl-to-meg-gpt2.slurm](./oscar-jsonl-to-meg-gpt2.slurm).

Runtime: about 12h

5. Final destination

We did all the processing on the SCRATCH partition which gets wiped out every 30 days, so we need to move the files to where they will not be deleted.

Final result:


Tar `oscar-shuffled.jsonl` and the dataset files to STORE:
