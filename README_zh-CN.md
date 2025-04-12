# doc page extractor

[English](./README.md) | 中文

## 简介

doc page extractor 可以将图片中的文字和格式加以识别，并返回结构化的数据。

## 安装

```shell
pip install doc-page-extractor
```

```shell
pip install onnxruntime==1.21.0
```

## 使用 CUDA

请参考 [PyTorch](https://pytorch.org/get-started/locally/) 的介绍，根据你的操作系统安装选择适当的命令安装。

此外，将前文安装 `onnxruntime` 的命令替换成如下：

```shell
pip install onnxruntime-gpu==1.21.0
```

## 例子

```python
from PIL import Image
from doc_page_extractor import DocExtractor


extractor = DocExtractor(
  model_dir_path=model_path, # AI 模型下载和安装的文件夹地址
  device="cpu", # 如果希望使用 CUDA，请改为 device="cuda:0" 这样的格式。
)
with Image.open("/path/to/your/image.png") as image:
  result = extractor.extract(
    image=image,
    lang="ch", # 图片文字的语言
  )
  for layout in result.layouts:
    for fragment in layout.fragments:
      print(fragment.rect, fragment.text)
```

## 致谢

本 repo 中 `doc_page_extractor/onnxocr` 的代码来自 [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)。

- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)
- [layoutreader](https://github.com/ppaanngggg/layoutreader)
- [StructEqTable](https://github.com/Alpha-Innovator/StructEqTable-Deploy)
- [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)