#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# to see https://github.com/opendatalab/DocLayout-YOLO/issues/93

source_file="./patches/tasks.py.patch"
target_file="./.venv/lib/python3.12/site-packages/doclayout_yolo/nn/tasks.py"

cp -f $source_file $target_file