#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

if [ -d "./doc_page_extractor/struct_eqtable" ]; then
  rm -rf "./doc_page_extractor/struct_eqtable"
fi
mkdir -p ./doc_page_extractor/struct_eqtable
curl -sL "https://github.com/Moskize91/StructEqTable/releases/download/v0.3.0.1/struct_eqtable.zip" | bsdtar -xzf - -C ./doc_page_extractor/struct_eqtable