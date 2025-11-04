import os
import torch
import tempfile

from typing import Literal
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from PIL import Image

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
    import flash_attn # type: ignore
    _ATTN_IMPLEMENTATION = "flash_attention_2"
except ImportError:
    _ATTN_IMPLEMENTATION = "eager"

class DeepSeekOCRModel:
    def __init__(self) -> None:
        model_name = "deepseek-ai/DeepSeek-OCR"
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            _attn_implementation=_ATTN_IMPLEMENTATION,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self._model = self._model.cuda().to(torch.bfloat16)

    def generate(self, image: Image.Image, prompt: str, size: DeepSeekOCRSize):
        config = _SIZE_CONFIGS[size]
        with tempfile.TemporaryDirectory() as temp_path:
            temp_image_path = os.path.join(temp_path, "temp_image.png")
            image.save(temp_image_path)
            text_result = self._model.infer(
                self._tokenizer,
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