# doc-page-extractor

Document page extraction tool that converts page images into text layouts with pixel coordinates.

The default backend remains local DeepSeek-OCR for existing users. Version 1.1 adds a unified OCR adapter layer with DeepSeek OpenAI-compatible Vendor support and Baidu cloud OCR support.

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

`Layout` keeps the original `ref`, `det`, and `text` fields. Version 1.1.1 also adds `kind`, a stable `LayoutKind` enum that callers should prefer over provider-specific labels. Adapter metadata remains available through optional fields such as `type`, `polygon`, `html`, `source`, and `raw`.

Use `extract_page_results()` when you need the structured page model:

```python
from doc_page_extractor import LayoutKind

for page_image, result in extractor.extract_page_results(
    image=Image.open("page.png"),
    size="gundam",
    stages=1,
    context=context,
):
    for block in result.structured.blocks:
        if block.kind == LayoutKind.TABLE:
            print(block.html)
```

The structured model groups asset captions with images, tables, and equations when possible. DeepSeek output is structured from flat OCR tags; Baidu Cloud OCR is normalized from Baidu's richer layout JSON into the same public kinds.

Baidu Cloud OCR extracts footnotes directly. If `stages > 1` is requested with the Baidu adapter, the extractor emits a warning and runs a single stage because DeepSeek-style multi-stage redaction can erase Baidu footnote regions.

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
