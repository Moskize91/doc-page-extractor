# doc-page-extractor

Document page extraction tool powered by DeepSeek-OCR.

## Installation

### Quick Start (CPU Version)

```bash
pip install doc-page-extractor
```

Or with Poetry:

```bash
poetry add doc-page-extractor
```

### GPU Support (Optional)

If you have an NVIDIA GPU and want faster inference, upgrade PyTorch to CUDA version after installation.

**For CUDA 12.1:**

With pip:
```bash
pip install doc-page-extractor
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

With Poetry:
```bash
poetry add doc-page-extractor
poetry add torch@latest torchvision@latest --source https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**

With pip:
```bash
pip install doc-page-extractor
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

With Poetry:
```bash
poetry add doc-page-extractor
poetry add torch@latest torchvision@latest --source https://download.pytorch.org/whl/cu118
```

## Usage

```python
from doc_page_extractor import main

# Your code here
```

## Verify Installation

Check if CUDA is available:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output for GPU:
```
PyTorch: 2.5.1+cu121
CUDA available: True
```

## Requirements

- Python >= 3.10, < 3.14
- For GPU: NVIDIA GPU with CUDA 11.8 or 12.1 support
- For CPU: Any system with sufficient RAM (slower inference)
