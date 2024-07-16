#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

cd build/test
./test_qr 1 16384 16384 -check
./test_potrf 1 16384 -check