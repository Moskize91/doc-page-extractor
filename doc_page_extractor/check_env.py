import warnings


_env_checked = False


def check_env() -> None:
    global _env_checked  # pylint: disable=global-statement
    if _env_checked:
        return
    _env_checked = True

    try:
        import torch
    except ImportError:
        warnings.warn(
            "This package requires PyTorch to run. Install it with: pip install torch torchvision",
            RuntimeWarning,
            stacklevel=2,
        )
        raise

    if torch.cuda.is_available():
        return
    warnings.warn(
        """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CUDA is not available!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This package requires CUDA to run, but torch.cuda.is_available() returned False.

  Possible causes:
  1. You installed CPU-only PyTorch. Reinstall with CUDA support:
     pip uninstall torch torchvision
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  2. Your NVIDIA GPU driver is outdated. Update it from:
     https://www.nvidia.com/download/index.aspx

  3. You don't have a CUDA-compatible GPU.

  To verify your setup, run: nvidia-smi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip(),
        RuntimeWarning,
        stacklevel=2,
    )
