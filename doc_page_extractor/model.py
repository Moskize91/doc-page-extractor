import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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
try:
    import flash_attn  # type: ignore # pylint: disable=unused-import

    _ATTN_IMPLEMENTATION = "flash_attention_2"
except ImportError:
    _ATTN_IMPLEMENTATION = "eager"

_Models = tuple[Any, Any]


class DeepSeekOCRModel:
    def __init__(self, model_path: Path | None, local_only: bool) -> None:
        self._model_name = "deepseek-ai/DeepSeek-OCR"
        self._cache_dir = str(model_path) if model_path else None
        self._local_only = local_only
        self._models: _Models | None = None

    def download(self) -> None:
        snapshot_download(
            repo_id=self._model_name,
            repo_type="model",
            cache_dir=self._cache_dir,
        )

    def load(self) -> None:
        self._ensure_models()

    def _ensure_models(self) -> _Models:
        if self._models is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                cache_dir=self._cache_dir,
                local_files_only=self._local_only,
            )
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                _attn_implementation=_ATTN_IMPLEMENTATION,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=self._cache_dir,
                local_files_only=self._local_only,
            )
            model = model.cuda().to(torch.bfloat16)
            self._models = (tokenizer, model)

        return self._models

    def generate(
        self, image: Image.Image, prompt: str, temp_path: str, size: DeepSeekOCRSize
    ) -> str:
        tokenizer, model = self._ensure_models()
        config = _SIZE_CONFIGS[size]
        temp_image_path = os.path.join(temp_path, "temp_image.png")
        image.save(temp_image_path)
        text_result = model.infer(
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
