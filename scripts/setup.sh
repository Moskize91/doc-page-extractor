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

conda create --prefix ./.venv python=3.12.7 -y

eval "$(conda shell.bash hook)"
conda activate ./.venv

pip install --upgrade pip
pip install -r requirements.txt

# to handle conflict hell: install paddleocr but ignore conflicting dependencies (recover dependencies list by hand)
pip install --no-deps paddleocr==2.9.1
pip install albucore==0.0.13 beautifulsoup4 Cython fonttools imgaug lmdb "numpy<2.0" \
            opencv-contrib-python opencv-python pillow pyclipper python-docx PyYAML RapidFuzz requests scikit-image shapely tqdm

scripts/patch.sh