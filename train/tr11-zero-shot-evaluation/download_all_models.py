from multiprocessing import Pool

from transformers import AutoModel

def _load_model(pretrain:str, revision: str):
    AutoModel.from_pretrained(pretrain, revision=revision)
    return f"Loaded: pretrain:{pretrain}, revision:{revision}"

def load_model(kwargs):
    return _load_model(**kwargs)

def main():
    pretrains = [
        "bigscience/tr3d-1B3-oscar-checkpoints",
        "bigscience/tr3e-1B3-c4-checkpoints",
        "bigscience/tr3m-1B3-pile-checkpoints"
    ]
    revisions = [
        "global_step19500",
        "global_step28500",
        "global_step37500",
        "global_step48000",
        "global_step57000",
        "global_step66000",
        "global_step76500",
        "global_step85500",
        "global_step94500",
        "global_step105000",
        "global_step114000",
    ]

    with Pool(10) as pool:
        results = pool.imap(
            load_model,
            [{"pretrain": pretrain, "revision": revision} for pretrain in pretrains for revision in revisions],
            chunksize=1
        )

        for result in results:
            print(result)

if __name__ == "__main__":
    main()
