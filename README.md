# doc page extractor

English | [中文](./README_zh-CN.md)

## Introduction

doc page extractor can identify text and format in images and return structured data.

## Installation

```shell
pip install doc-page-extractor
```

```shell
pip install onnxruntime==1.21.0
```

## Using CUDA

Please refer to the introduction of [PyTorch](https://pytorch.org/get-started/locally/) and select the appropriate command to install according to your operating system.

In addition, replace the command to install `onnxruntime` in the previous article with the following:

```shell
pip install onnxruntime-gpu==1.21.0
```

## Example

```python
from PIL import Image
from doc_page_extractor import DocExtractor

extractor = DocExtractor(
  model_dir_path=model_path, # Folder address where AI model is downloaded and installed
  device="cpu", # If you want to use CUDA, please change to device="cuda".
)
with Image.open("/path/to/your/image.png") as image:
  result = extractor.extract(
  image=image,
  lang="ch", # Language of image text
)
for layout in result.layouts:
  for fragment in layout.fragments:
    print(fragment.rect, fragment.text)
```

## Acknowledgements

The code of `doc_page_extractor/onnxocr` in this repo comes from [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR).

- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)
- [layoutreader](https://github.com/ppaanngggg/layoutreader)
- [StructEqTable](https://github.com/Alpha-Innovator/StructEqTable-Deploy)
- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)