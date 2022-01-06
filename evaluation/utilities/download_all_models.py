from argparse import ArgumentParser
from multiprocessing import Pool

from requests import HTTPError
from transformers import AutoModel, AutoTokenizer

def get_args():
    parser = ArgumentParser()
    # --experiments bigscience/tr3d-1B3-oscar-checkpoints,bigscience/tr3e-1B3-c4-checkpoints,bigscience/tr3m-1B3-pile-checkpoints
    parser.add_argument('--experiments', type=lambda s: s.split(','), required=True, help='Experiments we want to download.')
    # --steps 19500,28500,37500,48000,57000,66000,76500,85500,94500,105000,114000
    parser.add_argument('--steps', type=lambda s: [int(item) for item in s.split(',')], required=True, help='Steps we should download the model checkpoints')
    return parser.parse_args()

def _load_model(pretrain:str, revision: str):
    try:
        AutoModel.from_pretrained(pretrain, revision=revision)
        AutoTokenizer.from_pretrained(pretrain, revision=revision)
        return f"Loaded: {{pretrain:{pretrain}, revision:{revision}}}"
    except HTTPError:
        return f"Failed to load: {{pretrain:{pretrain}, revision:{revision}}}"

def load_model(kwargs):
    return _load_model(**kwargs)

def main():
    args = get_args()
    pretrains = args.experiments
    steps = args.steps
    revisions = [f"global_step{step}" for step in steps]

    # with Pool(10) as pool:
    #     results = pool.imap(
    #         load_model,
    #         [{"pretrain": pretrain, "revision": revision} for pretrain in pretrains for revision in revisions],
    #         chunksize=1
    #     )
    #
    #     for result in results:
    #         print(result)


    for kwargs in [{"pretrain": pretrain, "revision": revision} for pretrain in pretrains for revision in revisions]:
        print(load_model(kwargs))

if __name__ == "__main__":
    main()
