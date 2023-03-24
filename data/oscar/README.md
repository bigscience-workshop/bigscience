# OSCAR


## Megatron pre-processed files

These are the megatron-ready OSCAR files:

- Full 300M version (529GB) : `$six_ALL_CCFRWORK/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRWORK/datasets-custom/oscar-en-10k`

Each folder contains: `meg-gpt2_text_document.bin` and `meg-gpt2_text_document.idx` and Megatron-LM training script expects the following argument:
```
--data-path $six_ALL_CCFRWORK/datasets-custom/oscar-en/meg-gpt2_text_document
```

Should something get corrupted there is a backup:

- Full 300M version (529GB) : `$six_ALL_CCFRSTORE/datasets-custom/oscar-en`
- Tiny 10K version (56M): `$six_ALL_CCFRSTORE/datasets-custom/oscar-en-10k`




## How pre-processing was done

Here we used the original OSCAR 2019 release: https://oscar-project.org/post/oscar-2019/

In general the process is to first generate jsonl version of the dataset, while filtering out entries smaller than 1K, and then run that jsonl data through Megatron-LM preprocessing tool.

The rest of this document is the step by step process of accomplishing that in an efficient way.

**Update: Now that we better understand Megatron-LM's dataloader we know that it contacts all docs on the fly and delivers seqlen at a time as a single sample ([reference](https://github.com/NVIDIA/Megatron-LM/blob/90e0a0dd08159e1c95f4f9d99bb8687f327d36c3/megatron/data/gpt_dataset.py#L169-L185). So we don't need to filter out docs that are shorter than seqlen. Therefore in the future runs. We should adjust `oscar-to-jsonl.py` to remove the filtering.**

1. Convert `datasets` to `jsonl` which is the format required by Megatron-LM

The main script is [oscar-to-jsonl.py](./oscar-to-jsonl.py). Edit to change languages to use, initially using just English.

Note, that since shuffling slows the writeout process by 5-7 times, we don't shuffle in the script, but post-process it externally. See step 3.

To launch: [oscar-to-jsonl.slurm](./oscar-to-jsonl.slurm).

With "unshuffled_deduplicated_en" after filtering large entries (`>=1024`) we end up with 70754K examples out of 304230K total (about 1/4th of the full dataset).

The result is 5 files `oscar-[0-4].jsonl` of about 250GB each.

Runtime: 2-3h to download, ~2h to build, ~8h to filter, ~1.5h to write shards out


Update: `datasets` added multiproc `to_json` support:
https://github.com/huggingface/datasets/pull/2747
so it is in master, or after next after 1.11 version is released.


2. Concatenate

```
cat oscar-[0-4].jsonl > oscar-en.jsonl
```

This gives us a 1.2TB file.

Check:
```
$ wc -l oscar-en.jsonl
304230423 oscar-en.jsonl
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

Runtime: 2h



4. Megatron-LM preprocess

**Update**: that was an error, we can actually run for 100h on `-p cpu_p1` and so the normal script can complete no problem, but as a result of this mistake we can now pre-process data much faster.

We only have 20h to do processing which is not enough to process 300M records. Trying to do the whole thing in one preprocessing script took more than 24h and thus failed. Adding more than 16 workers didn't speed things up.

So we are splitting it in 4 chunks of ~80M records

```
split -l 77000000 oscar-en-shuffled.jsonl oscar
mv oscaraa oscar-en-shuffled-p1.jsonl
mv oscarab oscar-en-shuffled-p2.jsonl
mv oscarac oscar-en-shuffled-p3.jsonl
mv oscarad oscar-en-shuffled-p4.jsonl
```

We do the pre-processing:

The main script to launch: [oscar-jsonl-to-meg-gpt2.slurm](./oscar-jsonl-to-meg-gpt2.slurm), and we need to make copies of it for each chunk:

```
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-1.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-2.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-3.slurm
cp oscar-jsonl-to-meg-gpt2.slurm oscar-jsonl-to-meg-gpt2-4.slurm
perl -pi -e 's|p1|p1|' oscar-jsonl-to-meg-gpt2-1.slurm
perl -pi -e 's|p1|p2|' oscar-jsonl-to-meg-gpt2-2.slurm
perl -pi -e 's|p1|p3|' oscar-jsonl-to-meg-gpt2-3.slurm
perl -pi -e 's|p1|p4|' oscar-jsonl-to-meg-gpt2-4.slurm
```

```
sbatch oscar-jsonl-to-meg-gpt2-1.slurm
sbatch oscar-jsonl-to-meg-gpt2-2.slurm
sbatch oscar-jsonl-to-meg-gpt2-3.slurm
sbatch oscar-jsonl-to-meg-gpt2-4.slurm
```

This took about 6h each but run in parallel on different instances. This is surprisingly the projected time for the initial attempt to run in in one chunk, which was projected to 24 hours, and couldn't fit into 20h cap. So we finished the whole thing in 6 hours.

Outcome:

```
$ ls -1sh meg-gpt2-p*
131G meg-gpt2-p1_text_document.bin
1.4G meg-gpt2-p1_text_document.idx
131G meg-gpt2-p2_text_document.bin
1.4G meg-gpt2-p2_text_document.idx
131G meg-gpt2-p3_text_document.bin
1.4G meg-gpt2-p3_text_document.idx
138G meg-gpt2-p4_text_document.bin
1.5G meg-gpt2-p4_text_document.idx
```

Next merging: [oscar-meg-gpt2-merge.slurm](./oscar-meg-gpt2-merge.slurm)

Runtime: 22min - needed 26GB RSS RAM

Outcome: 304_230_423 records

```
$ ls -1sh meg-gpt2_text_document.*
529G meg-gpt2_text_document.bin
5.7G meg-gpt2_text_document.idx
```

Total runtime: under 7h.

Let's also make a small 10k version for experiments:

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

Tar/gz `oscar-shuffled.jsonl` and the dataset files to STORE, using [oscar-to-backup-tgz.slurm](./oscar-to-backup-tgz.slurm):

```
sbatch oscar-to-backup-tgz.slurm
```

6. Estimate total number of tokens

Make a 1GB slice:
```
$ head -79000 oscar-en-shuffled.jsonl > oscar-1GB.jsonl
$ ls -sh oscar-1GB.jsonl
1.0G oscar-1GB.jsonl
```

Analyze it (low mem-footprint):
```
$ python -c "import json, sys; \
from transformers import GPT2TokenizerFast; \
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2'); \
print(sum(tokenizer(json.loads(l)['text'], return_length=True).length[0] for l in sys.stdin.readlines()))" < oscar-1GB.jsonl
234260484
```

Extrapolate:

Thus 234M tokens in 1GB, ~280B tokens in 1.2TB (`234*1200`)

Incidentally this coincides with @Yozh's `FILE_SIZE_IN_GBS/4.5` formula! (average 4.5chars per word)
