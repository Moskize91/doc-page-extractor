# doc-page-extractor

Document page extraction tool that converts page images into text layouts with pixel coordinates.

The default backend remains local DeepSeek-OCR for existing users. Version 1.1 also adds a unified OCR adapter layer with DeepSeek OpenAI-compatible Vendor support and Baidu cloud OCR support.

## Installation

```bash
pip install doc-page-extractor
```

PyTorch is not installed automatically. You only need CUDA PyTorch when using the local DeepSeek-OCR backend.

## Backends

### Local DeepSeek-OCR

This is the default and keeps the existing API behavior:

```python
from doc_page_extractor import create_page_extractor

extractor = create_page_extractor()
```

Install CUDA PyTorch before using this backend:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check CUDA with:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### DeepSeek OCR Vendor

Use this backend when DeepSeek OCR is exposed through an OpenAI-compatible endpoint:

```python
from doc_page_extractor import (
    DeepSeekVendorOCRConfig,
    create_deepseek_vendor_page_extractor,
)

extractor = create_deepseek_vendor_page_extractor(
    DeepSeekVendorOCRConfig.from_env()
)
```

Expected environment variables:

```dotenv
DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_BASE_URL=
DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_API_KEY=
DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_MODEL=deepseek-ocr
DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_TEMPERATURE=0.0
DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_TOP_P=0.7
```

### Baidu Cloud OCR

Use this backend for Baidu Unlimited-OCR through Baidu Cloud:

```python
from doc_page_extractor import BaiduCloudOCRConfig, create_baidu_page_extractor

extractor = create_baidu_page_extractor(BaiduCloudOCRConfig.from_env())
```

Expected environment variables:

```dotenv
DOC_PAGE_EXTRACTOR_BAIDU_AK=
DOC_PAGE_EXTRACTOR_BAIDU_SK=
DOC_PAGE_EXTRACTOR_BAIDU_BASE_URL=https://aip.baidubce.com
```

## Extraction

All backends return the same `PageExtractor` shape:

```python
from PIL import Image
from doc_page_extractor import ExtractionContext

context = ExtractionContext(check_aborted=lambda: False)

for page_image, layouts in extractor.extract(
    image=Image.open("page.png"),
    size="gundam",
    stages=1,
    context=context,
):
    for layout in layouts:
        print(layout.det, layout.text)
```

`Layout` keeps the original `ref`, `det`, and `text` fields. Version 1.1 adds optional metadata fields such as `type`, `polygon`, `html`, `source`, and `raw` for adapters that provide richer layout data.

## Development

For contributors and developers, see [Development Guide](docs/DEVELOPMENT.md).

Useful local commands:

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
poetry run python scripts/ocr_sample.py --adapter both --image tests/images/friendly-title.png
```

## Requirements

- Python >= 3.10, < 3.14
- CUDA-capable NVIDIA GPU only when using local DeepSeek-OCR
- Remote OCR credentials only when using DeepSeek Vendor or Baidu Cloud OCR

## Dependencies & Licenses

This project is licensed under the MIT License. The local DeepSeek-OCR backend depends on the DeepSeek-OCR model, which uses **easydict** (LGPLv3) for configuration management.
