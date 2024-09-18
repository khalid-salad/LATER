#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

mkdir -p debug && cd debug
export CUDA_PATH=/usr/local/cuda

cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_80,code=sm_80" -DCUDA_ARCH="Ampere"
cmake --build .
