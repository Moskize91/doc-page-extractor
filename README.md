# doc-page-extractor

Document page extraction tool powered by DeepSeek-OCR.

## Installation

> **⚠️ Important:** This package requires PyTorch with CUDA support (GPU Required). PyTorch is NOT automatically installed - you must install it manually first.

### Step 1: Install PyTorch with CUDA

Choose the command that matches your CUDA version:

```bash
# For CUDA 12.1 (recommended for most users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> 💡 **Don't know your CUDA version?** Run `nvidia-smi` to check, or just try CUDA 12.1 (works with most recent drivers).

### Step 2: Install doc-page-extractor

```bash
pip install doc-page-extractor
```

### Verify Installation

Check if everything is working:

```bash
python -c "import doc_page_extractor; import torch; print('✓ Installation successful!'); print('✓ CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
✓ Installation successful!
✓ CUDA available: True
```

If CUDA shows `False`, see the troubleshooting section below.

## Usage

```python
from doc_page_extractor import PageExtractor

# Your code here
```

## Troubleshooting

### "PyTorch is required but not installed!"

Install PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA is not available!"

**Check your GPU driver:**
```bash
nvidia-smi
```

**If the command fails**, you need to install NVIDIA drivers:
- Download from: https://www.nvidia.com/download/index.aspx

**If it succeeds**, you might have CPU-only PyTorch. Reinstall with CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Requirements

- Python >= 3.10, < 3.14
- **NVIDIA GPU with CUDA 11.8 or 12.1 support (Required)**
- Sufficient GPU memory (recommended: 4GB+ VRAM)

## Dependencies & Licenses

This project is licensed under the MIT License. It depends on the DeepSeek-OCR model which uses **easydict** (LGPLv3) for configuration management.

## Development

For contributors and developers, see [Development Guide](docs/DEVELOPMENT.md) for:
- Running tests
- Running lint checks
- Building the package

