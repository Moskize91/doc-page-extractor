import os
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .abort import AbortContext
from .injection import InferWithInterruption

DeepSeekOCRSize = Literal["tiny", "small", "base", "large", "gundam"]


@dataclass
class _SizeConfig:
    base_size: int
    image_size: int
    crop_mode: bool


_SIZE_CONFIGS: dict[DeepSeekOCRSize, _SizeConfig] = {
    "tiny": _SizeConfig(base_size=512, image_size=512, crop_mode=False),
    "small": _SizeConfig(base_size=640, image_size=640, crop_mode=False),
    "base": _SizeConfig(base_size=1024, image_size=1024, crop_mode=False),
    "large": _SizeConfig(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": _SizeConfig(base_size=1024, image_size=640, crop_mode=True),
}

_ATTN_IMPLEMENTATION: str
if find_spec("flash_attn") is not None:
    _ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    _ATTN_IMPLEMENTATION = "eager"

_Models = tuple[Any, Any]


class DeepSeekOCRModel:
    def __init__(self, model_path: Path | None, local_only: bool) -> None:
        if local_only and model_path is None:
            raise ValueError("model_path must be provided when local_only is True")

        self._lock: Lock = Lock()
        self._model_name = "deepseek-ai/DeepSeek-OCR"
        self._model_path: Path | None = model_path
        self._local_only = local_only
        self._models: _Models | None = None

    def download(self) -> None:
        with self._lock:
            snapshot_download(
                repo_id=self._model_name,
                repo_type="model",
                cache_dir=self._cache_dir(),
            )
            if self._model_path is not None and self._find_pretrained_path() is None:
                raise RuntimeError(
                    f"Model downloaded but not found in expected cache structure. "
                    f"Expected path: {self._model_path}/models--deepseek-ai--DeepSeek-OCR/snapshots/. "
                    f"This may indicate a Hugging Face cache structure change. "
                    f"Please report this issue."
                )

    def load(self) -> None:
        with self._lock:
            self._ensure_models()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        temp_path: str,
        size: DeepSeekOCRSize,
        aborted_context: AbortContext | None,
    ) -> str:
        with self._lock:
            tokenizer, model = self._ensure_models()
            config = _SIZE_CONFIGS[size]
            temp_image_path = os.path.join(temp_path, "temp_image.png")
            image.save(temp_image_path)
            with InferWithInterruption(
                model=model,
                aborted_context=aborted_context,
            ) as infer:
                text_result = infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=temp_path,
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    save_results=True,
                    test_compress=True,
                    eval_mode=True,
                )
            return text_result

    def _ensure_models(self) -> _Models:
        if self._models is not None:
            return self._models

        name_or_path = self._model_name
        cache_dir: str | None = None

        if self._local_only:
            name_or_path = self._find_pretrained_path()
            if name_or_path is None:
                raise ValueError(
                    f"Local model not found at {self._model_path}. "
                    f"Expected Hugging Face cache structure: "
                    f"{self._model_path}/models--deepseek-ai--DeepSeek-OCR/snapshots/[hash]/. "
                    f"Please run download_models() first to download the model."
                )
        else:
            cache_dir = self._cache_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=self._local_only,
        )
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            _attn_implementation=_ATTN_IMPLEMENTATION,
            trust_remote_code=True,
            use_safetensors=True,
            cache_dir=cache_dir,
            local_files_only=self._local_only,
        )
        model = model.cuda().to(torch.bfloat16)
        self._models = (tokenizer, model)

        return self._models

    def _cache_dir(self) -> str | None:
        if self._model_path is not None:
            return str(self._model_path)
        return None

    def _find_pretrained_path(self) -> str | None:
        # Hugging Face 缓存结构: cache_dir/models--{org}--{model}/snapshots/{hash}/
        assert self._model_path is not None
        cache_model_dir = self._model_path / "models--deepseek-ai--DeepSeek-OCR"
        if not cache_model_dir.exists():
            return None

        ref_file = cache_model_dir / "refs" / "main"
        if ref_file.exists() and ref_file.is_file():
            snapshot_hash = ref_file.read_text().strip()
            snapshot_path = cache_model_dir / "snapshots" / snapshot_hash
            if snapshot_path.exists() and snapshot_path.is_dir():
                return str(snapshot_path)

        snapshots_dir = cache_model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            return None
        latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
        return str(latest_snapshot)
