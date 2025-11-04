"""Document page extraction tool powered by DeepSeek-OCR.

This package requires PyTorch with CUDA support. Please install PyTorch before using:

For CUDA 12.1:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

For CUDA 11.8:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

For more information, see: https://pytorch.org/get-started/locally/
"""

# Check PyTorch availability on import
try:
    import torch
    import torchvision
except ImportError as e:
    raise ImportError(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  PyTorch is required but not installed!\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "  Please install PyTorch with CUDA support before using this package:\n"
        "\n"
        "  For CUDA 12.1 (recommended):\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "\n"
        "  For CUDA 11.8:\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
        "\n"
        "  For more options, visit: https://pytorch.org/get-started/locally/\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ) from e

# Check CUDA availability
if not torch.cuda.is_available():
    import warnings
    warnings.warn(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  CUDA is not available!\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "  This package requires CUDA to run, but torch.cuda.is_available() returned False.\n"
        "\n"
        "  Possible causes:\n"
        "  1. You installed CPU-only PyTorch. Reinstall with CUDA support:\n"
        "     pip uninstall torch torchvision\n"
        "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
        "\n"
        "  2. Your NVIDIA GPU driver is outdated. Update it from:\n"
        "     https://www.nvidia.com/download/index.aspx\n"
        "\n"
        "  3. You don't have a CUDA-compatible GPU.\n"
        "\n"
        "  To verify your setup, run: nvidia-smi\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        RuntimeWarning,
        stacklevel=2
    )

from .model import DeepSeekOCRSize
from .extractor import PageExtractor
from .plot import plot

__version__ = "1.0.0"
__all__ = ["DeepSeekOCRSize", "PageExtractor", "plot"]