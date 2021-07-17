# OSCAR


## Megatron pre-processed files

These are the megatron-ready OSCAR files:

- Full 70754K version (393GB) : `$six_ALL_CCFRWORK/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k`

Each folder contains: `meg-gpt2_text_document.bin` and `meg-gpt2_text_document.idx` and Megatron-LM training script expects the following argument:
```
--data-path $six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document
```

Should something get corrupted there is a backup:

- Full 70754K version (393GB) : `$six_ALL_CCFRSTORE/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRSTORE/datasets-custom/oscar-en-10k`




## How pre-processing was done

In general the process is to first generate jsonl version of the dataset, while filtering out entries smaller than 1K, and then run that jsonl data through Megatron-LM preprocessing tool.

The rest of this document is the step by step process of accomplishing that in an efficient way.

1. Convert `datasets` to `jsonl` which is the format required by Megatron-LM

The main script is [oscar-to-jsonl.py](./oscar-to-jsonl.py). Edit to change languages to use, initially using just English.

Note, that since shuffling slows the writeout process by 5-7 times, we don't shuffle in the script, but post-process it externally. See step 3.

To launch: [oscar-to-jsonl.slurm](./oscar-to-jsonl.slurm).

With "unshuffled_deduplicated_en" after filtering large entries (`>=1024`) we end up with 70754K examples out of 304230K total (about 1/4th of the full dataset).

The result is 5 files `oscar-[0-4].jsonl` of about 180GB each.

Runtime: 2-3h to download, ~2h to build, ~8h to filter, ~1.5h to write shards out


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

Outcome:

```
ls -sh *
392G meg-gpt2_text_document.bin
1.4G meg-gpt2_text_document.idx
```


Runtime: about 13h

Let's make a small 10k version for experiments:

```
head -10000 oscar-shuffled.jsonl > oscar-shuffled-10k.jsonl
```
and then process with the same slurm script above, but changing the input to `oscar-shuffled-10k.jsonl`



5. Final destination

We did all the processing on the SCRATCH partition which gets wiped out every 30 days, so we need to move the files to where they will not be deleted.

Since at this moment we used just the English part of the OSCAR dataset, let's include that in the folder name to differentiate from other builds that will be multi-lingual.

Make the final result which will be used by the megatron training script available on the persistent WORK partition:

```
mkdir oscar-en
mv meg-gpt2_text_document.* oscar-en
cp -r oscar-en $six_ALL_CCFRWORK/datasets-custom
```

Back it up to STORE:

It's already binary and just 2 files, so no need to tar (STORE has limited inodes)
```
mkdir -p $six_ALL_CCFRSTORE/datasets-custom
cp -r oscar-en $six_ALL_CCFRSTORE/datasets-custom
```

Also copy the small version for experiments to WORK and STORE:
```
cp -r oscar-en-10k $six_ALL_CCFRWORK/datasets-custom
cp -r oscar-en-10k $six_ALL_CCFRSTORE/datasets-custom
```

Tar/gz `oscar-shuffled.jsonl` and the dataset files to STORE:

```


```
