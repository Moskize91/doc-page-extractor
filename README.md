# doc page extractor

English | [中文](./README_zh-CN.md)

## Introduction

doc page extractor can identify text and format in images and return structured data.

## Installation

```shell
pip install doc-page-extractor
```

## Using CUDA

Please refer to the introduction of [PyTorch](https://pytorch.org/get-started/locally/) and select the appropriate command to install according to your operating system.

## Example

```python
from PIL import Image
from doc_page_extractor import DocExtractor

extractor = DocExtractor(
  model_dir_path=model_path, # Folder address where AI model is downloaded and installed
  device="cpu", # If you want to use CUDA, please change to device="cuda:0".
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

- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [layoutreader](https://github.com/ppaanngggg/layoutreader)