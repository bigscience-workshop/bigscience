#!/usr/bin/env python

# creates a local auth token file which can then be safely used by other programs without leaking
# the password in public git

import getpass
import json
from pathlib import Path
from huggingface_hub import HfApi

HUB_DATA_PATH = Path(__file__).parent / ".hub_info.json"

username = input("Hub username: ")
password = getpass.getpass("Hub password: ")
email = input("Hub email: ")
auth_token = HfApi().login(username=username, password=password)

data = dict(username=username, email=email, auth_token=auth_token)
#print(data)

with open(HUB_DATA_PATH, 'w') as f:
    json.dump(data, f)
