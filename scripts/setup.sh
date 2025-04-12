#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

if ! command -v conda >/dev/null 2>&1; then
  echo "Need Conda to setup the environment"
  exit 1
fi

if [ -d ".venv" ]; then
  rm -rf .venv
fi

conda create --prefix ./.venv python=3.10.16 -y

eval "$(conda shell.bash hook)"
conda activate ./.venv

pip install --upgrade pip
pip install -r requirements.txt
pip install onnxruntime==1.21.0

if [ -d "./struct_eqtable" ]; then
  rm -rf "./struct_eqtable"
fi
mkdir -p ./struct_eqtable
curl -sL "https://github.com/Moskize91/StructEqTable/releases/download/v0.3.0.1/struct_eqtable.zip" | tar -xz -C ./struct_eqtable