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
import io
import json
import os
import re
import subprocess
import sys

from collections import defaultdict
from fnmatch import fnmatch
from huggingface_hub import HfApi, HfFolder, Repository
from pathlib import Path
from typing import List, Optional, Union

HUB_DATA_PATH = Path(__file__).resolve().parent / ".hub_info.json"

# map https://git-scm.com/docs/git-status#_short_format
#

#     ' ' = unmodified
#     M = modified
#     A = added
#     D = deleted
#     R = renamed
#     C = copied
#     U = updated but unmerged

# X          Y     Meaning
# -------------------------------------------------
# 	 [AMD]   not updated
# M        [ MD]   updated in index
# A        [ MD]   added to index
# D                deleted from index
# R        [ MD]   renamed in index
# C        [ MD]   copied in index
# [MARC]           index and work tree matches
# [ MARC]     M    work tree changed since index
# [ MARC]     D    deleted in work tree
# [ D]        R    renamed in work tree
# [ D]        C    copied in work tree
# -------------------------------------------------
# D           D    unmerged, both deleted
# A           U    unmerged, added by us
# U           D    unmerged, deleted by them
# U           A    unmerged, added by them
# D           U    unmerged, deleted by us
# A           A    unmerged, both added
# U           U    unmerged, both modified
# -------------------------------------------------
# ?           ?    untracked
# !           !    ignored

git_status_lookup = {
    "?": "untracked",
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "updated_unmerged",
}

def get_git_files_by_status(local_dir):
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
        return {}

    file_statuses = [status.strip() for status in git_status.split("\n")]

    # create a dict of lists for each long key in git_status_lookup
    files = defaultdict(list)
    for l in file_statuses:
        k, v = l.split(' ', 1)
        k = k.strip()[0] # get first column
        # remap to sensible name
        k = git_status_lookup.get(k, "unknown")
        files[k].append(v)

    #print(files)

    return files


# XXX: this should be PR'ed into https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/repository.py
# after adjusting the API self, self.local_dir
def get_untracked_files(local_dir) -> List[str]:
    """
    Returns a list of untracked files in the working directory
    """
    key = "untracked"
    files_by_status = get_git_files_by_status(local_dir)
    return files_by_status[key] if key in files_by_status else []

def get_modified_files(local_dir) -> List[str]:
    """
    Returns a list of untracked files in the working directory
    """
    key = "modified"
    files_by_status = get_git_files_by_status(local_dir)
    return files_by_status[key] if key in files_by_status else []

def run_cmd(cmd, local_dir):
    try:
        git_status = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=local_dir,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return git_status


def hub_config_repo(hub_data, local_dir):

    # if we have the bot user email set, that means we have done this process already
    # but some users don't have any `user.email` set, so recover gracefully if that's the case
    try:
        cmd = f"git config user.email"
        email = run_cmd(cmd.split(), local_dir)
        if len(email) > 0 and email == hub_data['email']:
            return
    except:
        pass

    print(f"* Detected a new clone. Setting it up for {hub_data['username']}")

    # to work as another user we need
    # 1. their user.email
    cmd = f"git config user.email {hub_data['email']}"
    run_cmd(cmd.split(), local_dir)

    # 2. pre-auth the repo
    # a. get url
    cmd = "git remote get-url origin"
    url = run_cmd(cmd.split(), local_dir)

    # b. extract just the huggingface.co/app-test-user/test-tensorboard part
    repo_part_url = re.sub(r'https.*(?=huggingface)', '', url, 0, re.M)
    cmd = f"git remote set-url origin --push https://{hub_data['username']}:{hub_data['auth_token']}@{repo_part_url}"
    run_cmd(cmd.split(), local_dir)


def get_hub_data():
    hub_data_path = "./.hub_info.json"
    if not os.path.isfile(HUB_DATA_PATH):
        raise FileNotFoundError(f"File '{hub_data_path}' doesn't exist. Please run hub-auth.py first")

    with io.open(HUB_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

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

    hub_data = get_hub_data()
    repo = Repository(args.repo_path)#, use_auth_token=use_auth_token)

    hub_config_repo(hub_data, local_dir=args.repo_path)

    files_dict = get_git_files_by_status(args.repo_path)
    #from pprint import pprint
    #pprint(files_dict)
    #sys.exit()

    # we want untracked and modified files
    uncommitted_files = get_untracked_files(args.repo_path)
    uncommitted_files.extend(get_modified_files(args.repo_path))
    print(f"* Found {len(uncommitted_files)} uncommitted files:")
    if args.debug:
        print(''.join(f"- {f}\n" for f in uncommitted_files))

    total_to_commit = 0
    for pattern in args.patterns:

        # *** new and modified files ***
        # check that these are the files that match the pattern passed to git_add
        uncommitted_files_matched = [f for f in uncommitted_files if fnmatch(f, pattern)]
        print(f"* Found {len(uncommitted_files_matched)} uncommitted files matching pattern: {pattern}:")

        if args.debug:
            print(''.join(f"- {f}\n" for f in uncommitted_files_matched))

        if len(uncommitted_files_matched) > 0:
            total_to_commit += len(uncommitted_files_matched)

            # # auto_lfs_track requires huggingface-hub-0.0.15, but transformers forces 0.0.12
            repo.git_add(pattern=pattern) # , auto_lfs_track=True)
            repo.git_commit(commit_message="new data")

    if total_to_commit:
        print(f"* Pushing {total_to_commit} files")
        repo.git_push()
        print("* Pushed")
    else:
        print("* Detected no new or modified files. Nothing to push.")




if __name__ == "__main__":

    main()
