#!/usr/bin/env python

# creates a local auth token file which can then be safely used by other programs without leaking
# the password in public git

import getpass
from huggingface_hub import HfApi

username = input("Hub username: ")
password = getpass.getpass("Hub password: ")

print(username, password)
use_auth_token = HfApi().login(username=username, password=password)

path = "./.auth_token"
with open(path, "w+") as f:
    f.write(use_auth_token)
