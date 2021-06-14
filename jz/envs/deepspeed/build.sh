#!/bin/bash

rm -rf build

time TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log

# time TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_OPS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
