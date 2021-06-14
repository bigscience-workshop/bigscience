#!/bin/bash

pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log

