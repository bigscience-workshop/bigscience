#!/usr/bin/env python

#
# this tool automatically pushes newly added files into the hub repo
#
# If the program fails to run the first time make sure to run `hub-auth.py` to authenticate and save
# the token locally which will then be used by this program
#
# Example:
#
# tools/files2hub.py --repo-path /hf/Megatron-DeepSpeed-master/output_dir/tensorboard/ --patterns '*tfevents*' -d
#
# multiple patterns can be passed

import argparse
import os
import subprocess
from fnmatch import fnmatch
from huggingface_hub import HfApi, HfFolder, Repository
from typing import List, Optional, Union


# XXX: this should be PR'ed into https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repository.py
# after adjusting the API self, self.local_dir
def get_untracked_files(local_dir) -> List[str]:
    """
    Returns a list of untracked files in the working directory
    """
    try:
        git_status = subprocess.run(
            ["git", "status", "-s"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=local_dir,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    if len(git_status) == 0:
        return []

    file_statuses = [status.strip() for status in git_status.split("\n")]

    # Only keep files that are untracked using the ?? prefix
    untracked_file_statuses = [
        status for status in file_statuses if "??" in status.split()[0]
    ]

    # Remove the D prefix and strip to keep only the relevant filename
    untracked_files = [
        status.split()[-1].strip() for status in untracked_file_statuses
    ]

    #print(untracked_files)

    return untracked_files

def get_auth_token():
    auth_token_path = "./.auth_token"
    if not os.path.isfile(auth_token_path):
        raise FileNotFoundError(f"File '{auth_token_path}' doesn't exist. Please run hub-auth.py first")

    token = ""
    with open(auth_token_path, "r") as f:
        token = f.read()

    if len(token) == 0:
        raise ValueError(f"File '{auth_token_path}' is empty. Please run hub-auth.py first")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs='+', default=None, required=True, type=str, help="one or more patterns of files to match to add to the hub - make sure to quote those!")
    parser.add_argument("--repo-path", type=str, required=True, help="path to the already cloned repo")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    return parser.parse_args()

def main():

    args = get_args()

    if not os.path.isdir(args.repo_path):
        raise FileNotFoundError(f"Repository directory '{args.repo_path}' doesn't exist. Have you cloned the repo")

    if len(args.patterns) == 0:
        raise ValueError("at least one --pattern is required")

    if args.debug:
        print(f"Tracking {len(args.patterns)} patterns:")
        print(''.join(f"- {x}\n" for x in args.patterns))

    use_auth_token = get_auth_token()
    repo = Repository(args.repo_path, use_auth_token=use_auth_token)

    untracked_files = get_untracked_files(args.repo_path)
    print(f"* Found {len(untracked_files)} untracked files:")
    if args.debug:
        print(''.join(f"- {f}\n" for f in untracked_files))

    need_to_push = False
    for pattern in args.patterns:

        # check that these are the files that match the pattern passed to git_add
        untracked_files_matched = [f for f in untracked_files if fnmatch(f, pattern)]
        print(f"* Found {len(untracked_files_matched)} untracked files matching pattern: {pattern}:")
        if args.debug:
            print(''.join(f"- {f}\n" for f in untracked_files_matched))

        if len(untracked_files_matched) > 0:
            need_to_push = True

            # auto_lfs_track requires huggingface-hub-0.0.15, but transformers forces 0.0.12
            #repo.git_add(pattern=pattern, auto_lfs_track=True)
            repo.git_add(pattern=pattern)

            repo.git_commit(commit_message="test")

            #except Exception:
            #    print("probably nothing was added?")


    if need_to_push:
        print("Pushing")
        repo.git_push()
        print("Pushed")
    else:
        print("No new files were added to git.")




if __name__ == "__main__":

    main()
