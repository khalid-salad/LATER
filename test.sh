#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

cd debug/test
# ./test_qr 1 16384 16384 -check
# ./test_syrk 16384 16384 -check
./test_potrf 134672 -check