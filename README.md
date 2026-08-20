# doc-page-extractor

Document page extraction tool that converts page images into text layouts with pixel coordinates.

The package provides local Hugging Face OCR backends and vendor OCR adapters
that all return the same page layout shape.

## Installation

```bash
pip install doc-page-extractor
```

PyTorch is not installed automatically. You only need CUDA PyTorch when using a
local Hugging Face OCR backend.

## Backends

### DeepSeek OCR Local

Use this backend for local DeepSeek OCR or DeepSeek OCR 2 inference:

```python
from doc_page_extractor import create_deepseek_ocr_page_extractor

extractor = create_deepseek_ocr_page_extractor(
    ocr_model="deepseek-ocr",
    model_path="models-cache",
    local_only=True,
)

extractor2 = create_deepseek_ocr_page_extractor(
    ocr_model="deepseek-ocr2",
    model_path="models-cache",
    local_only=True,
)
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

### Unlimited OCR Local

Use this backend for local Unlimited OCR Transformers inference:

```python
from doc_page_extractor import create_unlimited_ocr_page_extractor

extractor = create_unlimited_ocr_page_extractor(
    model_path="models-cache",
    local_only=True,
)
```

The local Unlimited OCR backend uses the Hugging Face model
`baidu/Unlimited-OCR`. Single-page local inference supports the `base` and
`gundam` size presets.

### DeepSeek OCR Vendor

Use this backend when DeepSeek OCR is exposed through an OpenAI-compatible endpoint:

```python
from doc_page_extractor import (
    DeepSeekOCRVendorConfig,
    create_deepseek_ocr_vendor_page_extractor,
)

extractor = create_deepseek_ocr_vendor_page_extractor(
    DeepSeekOCRVendorConfig(
        base_url="https://example.test/openai",
        api_key="...",
        model="deepseek-ocr",
    )
)
```

The package does not read environment variables automatically. `.env.template`
is only for local debugging scripts.

```dotenv
DEEPSEEK_OCR_BASE_URL=
DEEPSEEK_OCR_API_KEY=
DEEPSEEK_OCR_MODEL=deepseek-ocr
DEEPSEEK_OCR_TEMPERATURE=0.0
DEEPSEEK_OCR_TOP_P=0.7
DEEPSEEK_OCR_MAX_TOKENS=8000
DEEPSEEK_OCR_TIMEOUT_SECONDS=180
```

### DeepSeek OCR 2 Vendor

Use this backend for DeepSeek OCR 2 through an OpenAI-compatible endpoint:

```python
from doc_page_extractor import (
    DeepSeekOCR2VendorConfig,
    create_deepseek_ocr2_vendor_page_extractor,
)

extractor = create_deepseek_ocr2_vendor_page_extractor(
    DeepSeekOCR2VendorConfig(
        base_url="https://example.test/openai",
        api_key="...",
        model="deepseek-ocr2",
    )
)
```

The package does not read environment variables automatically. `.env.template`
is only for local debugging scripts.

```dotenv
DEEPSEEK_OCR2_BASE_URL=
DEEPSEEK_OCR2_API_KEY=
DEEPSEEK_OCR2_MODEL=
DEEPSEEK_OCR2_TEMPERATURE=0.0
DEEPSEEK_OCR2_TOP_P=0.7
DEEPSEEK_OCR2_MAX_TOKENS=8000
DEEPSEEK_OCR2_TIMEOUT_SECONDS=180
```

### Unlimited OCR Vendor

Use this backend for Baidu Cloud Unlimited OCR:

```python
from doc_page_extractor import (
    UnlimitedOCRVendorConfig,
    create_unlimited_ocr_vendor_page_extractor,
)

extractor = create_unlimited_ocr_vendor_page_extractor(
    UnlimitedOCRVendorConfig(
        ak="...",
        sk="...",
    )
)
```

The package does not read environment variables automatically. `.env.template`
is only for local debugging scripts.

```dotenv
UNLIMITED_OCR_ACCESS_KEY=
UNLIMITED_OCR_SECRET_KEY=
UNLIMITED_OCR_BASE_URL=https://aip.baidubce.com
UNLIMITED_OCR_POLL_INTERVAL_SECONDS=2
UNLIMITED_OCR_TIMEOUT_SECONDS=180
```

Unlimited OCR Vendor images with a side longer than 8192 px are resized
proportionally before upload. Returned layout coordinates are mapped back to the
original image size.

## Extraction

All backends return the same `PageExtractor` shape:

```python
from PIL import Image
from doc_page_extractor import ExtractionContext

context = ExtractionContext(check_aborted=lambda: False)

for page_image, result in extractor.extract_page_results(
    image=Image.open("page.png"),
    size="gundam",
    stages=1,
    context=context,
):
    for layout in result.layouts:
        print(layout.kind, layout.det, layout.text)
```

`Layout.kind` is the stable layout semantic. Adapter metadata remains available through optional fields such as `type`, `polygon`, `html`, `source`, and `raw`.

Structured page blocks are available on each `OCRPageResult`:

```python
from doc_page_extractor import LayoutKind

for page_image, result in extractor.extract_page_results(
    image=Image.open("page.png"),
    size="gundam",
    stages=1,
    context=context,
):
    if result.structured is None:
        continue
    for block in result.structured.blocks:
        if block.kind == LayoutKind.TABLE:
            print(block.html)
```

The structured model groups asset captions with images, tables, and equations
when possible. DeepSeek output is structured from flat OCR tags; DeepSeek OCR 2
output is structured from line blocks; Unlimited OCR local output is structured
from local detection tags; Unlimited OCR Vendor output is normalized from richer
layout JSON into the same public kinds.

Unlimited OCR extracts footnotes directly. If `stages > 1` is requested with an
Unlimited OCR adapter, the extractor emits a warning and runs a single stage
because DeepSeek-style multi-stage redaction can erase footnote regions.

## Development

For contributors and developers, see [Development Guide](docs/DEVELOPMENT.md).

Useful local commands:

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
poetry run python scripts/ocr_sample.py --adapter unlimited-ocr-vendor --image tests/images/friendly-title.png
```

## Requirements

- Python >= 3.10, < 3.14
- CUDA-capable NVIDIA GPU only when using local Hugging Face OCR backends
- Remote OCR credentials only when using vendor OCR backends

## Dependencies & Licenses

This project is licensed under the MIT License. Local Hugging Face OCR backends
depend on their upstream model code and runtime dependencies. The DeepSeek-OCR
model uses **easydict** (LGPLv3) for configuration management.
