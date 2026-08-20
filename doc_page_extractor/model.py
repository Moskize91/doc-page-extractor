from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
from readerwriterlock import rwlock
from transformers import AutoModel, AutoTokenizer

from .types import DeepSeekOCRSize
from .check_env import check_env
from .extraction_context import ExtractionContext
from .injection import InferWithInterruption, preprocess_model


@dataclass
class _SizeConfig:
    base_size: int
    image_size: int
    crop_mode: bool


_DEEPSEEK_OCR_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"
_DEEPSEEK_OCR2_REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
_UNLIMITED_OCR_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"

_DEEPSEEK_SIZE_CONFIGS: dict[DeepSeekOCRSize, _SizeConfig] = {
    "tiny": _SizeConfig(base_size=512, image_size=512, crop_mode=False),
    "small": _SizeConfig(base_size=640, image_size=640, crop_mode=False),
    "base": _SizeConfig(base_size=1024, image_size=1024, crop_mode=False),
    "large": _SizeConfig(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": _SizeConfig(base_size=1024, image_size=640, crop_mode=True),
}

_UNLIMITED_SIZE_CONFIGS: dict[DeepSeekOCRSize, _SizeConfig] = {
    "base": _SizeConfig(base_size=1024, image_size=1024, crop_mode=False),
    "gundam": _SizeConfig(base_size=1024, image_size=640, crop_mode=True),
}

_ATTN_IMPLEMENTATION: str
if find_spec("flash_attn") is not None:
    _ATTN_IMPLEMENTATION = "flash_attention_2"
else:
    _ATTN_IMPLEMENTATION = "eager"


@dataclass
class _Models:
    tokenizer: AutoTokenizer
    llms: list[AutoModel]


class HuggingFaceBackend:
    def __init__(
        self,
        model_name: str,
        model_path: Path | None,
        local_only: bool,
        enable_devices_numbers: Iterable[int] | None,
        default_revision: str,
        attn_implementation: str | None = None,
    ) -> None:
        if local_only and model_path is None:
            raise ValueError(
                "model_path must be provided when local_only is True")

        self._rwlock = rwlock.RWLockFair()
        self._model_name = model_name
        self._default_revision = default_revision
        self._model_path: Path | None = model_path
        self._local_only = local_only
        self._models: _Models | None = None
        self._enable_devices_numbers: Iterable[int] | None = enable_devices_numbers
        self._device_number_to_index: list[int | None] | None = None
        self._attn_implementation = attn_implementation or _ATTN_IMPLEMENTATION

    def download(self, revision: str | None) -> None:
        with self._rwlock.gen_wlock():
            snapshot_download(
                repo_id=self._model_name,
                repo_type="model",
                revision=revision or self._default_revision,
                # Keep incomplete blobs so interrupted large downloads can resume.
                force_download=False,
                cache_dir=self._cache_dir(),
            )
            if self._model_path is not None and self._find_pretrained_path() is None:
                raise RuntimeError(
                    f"Model downloaded but not found in expected cache structure. "
                    f"Expected path: {self._cache_model_dir()}/snapshots/. "
                    f"This may indicate a Hugging Face cache structure change. "
                    f"Please report this issue."
                )

    def load(self) -> None:
        self._ensure_models()

    def unload(self) -> None:
        with self._rwlock.gen_wlock():
            if self._models is not None:
                self._models = None

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> str:

        tokenizer, llm_model = self._select_model(device_number)
        config = _DEEPSEEK_SIZE_CONFIGS[size]

        with self._rwlock.gen_rlock():
            with InferWithInterruption(llm_model, context) as infer:
                # - {output_path}/result.mmd - OCR提取的Markdown格式结果
                # - {output_path}/result_with_boxes.jpg - 带有边界框标注的可视化图片
                # - {output_path}/images/{N}.jpg - 从文档中提取的图片（N为索引号）
                # - {output_path}/geo.jpg - 如果检测到几何图形会生成该文件（条件性）
                text_result = infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(output_path),
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    save_results=True,
                    test_compress=True,
                    eval_mode=True,
                )
            return text_result

    def _select_model(self, device_number: int | None):
        models = self._ensure_models()
        if device_number is None:
            model_index = self._get_device_number_to_index()[0]
        else:
            model_index = self._get_device_number_to_index()[device_number]

        if model_index is None:
            raise ValueError(f"Device number {device_number} is not enabled.")

        return models.tokenizer, models.llms[model_index]

    def _ensure_models(self) -> _Models:
        check_env()
        import torch

        with self._rwlock.gen_rlock():
            if self._models is not None:
                return self._models

        with self._rwlock.gen_wlock():
            # 检查两次，因为中间有解锁
            if self._models is not None:
                return self._models

            device_number_to_index = self._get_device_number_to_index()
            if len(device_number_to_index) == 0:
                raise RuntimeError("No CUDA devices available")

            name_or_path = self._model_name
            cache_dir: str | None = None

            if self._local_only:
                name_or_path = self._find_pretrained_path()
                if name_or_path is None:
                    raise ValueError(
                        f"Local model not found at {self._model_path}. "
                        f"Expected Hugging Face cache structure: "
                        f"{self._cache_model_dir()}/snapshots/[hash]/. "
                        f"Please run download_ocr_model() first to download the model."
                    )
            else:
                cache_dir = self._cache_dir()

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=name_or_path,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=self._local_only,
            )
            llm_models: list[AutoModel] = []
            for device_number, model_index in enumerate(device_number_to_index):
                if model_index is None:
                    continue
                model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path=name_or_path,
                    _attn_implementation=self._attn_implementation,
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                    local_files_only=self._local_only,
                )
                model = model.to(torch.bfloat16).cuda(device_number)
                llm_models.append(preprocess_model(model))

            self._models = _Models(
                tokenizer=tokenizer,
                llms=llm_models,
            )
            return self._models

    def _cache_dir(self) -> str | None:
        if self._model_path is not None:
            return str(self._model_path)
        return None

    def _find_pretrained_path(self) -> str | None:
        # Hugging Face 缓存结构: cache_dir/models--{org}--{model}/snapshots/{hash}/
        assert self._model_path is not None
        cache_model_dir = self._cache_model_dir()
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

    def _cache_model_dir(self) -> Path:
        assert self._model_path is not None
        model_dir_name = f"models--{self._model_name.replace('/', '--')}"
        return self._model_path / model_dir_name

    def _get_device_number_to_index(self) -> list[int | None]:
        if self._device_number_to_index is None:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if self._enable_devices_numbers is None:
                    self._device_number_to_index = list(range(device_count))
                else:
                    next_model_index: int = 0
                    device_number_to_index: list[int | None] = [
                        None] * device_count
                    for enable_device_number in sorted(list(set(self._enable_devices_numbers))):
                        if enable_device_number < 0 or enable_device_number >= device_count:
                            raise ValueError(
                                f"Invalid device number {enable_device_number}, "
                                f"your system has {device_count} CUDA devices."
                            )
                        device_number_to_index[enable_device_number] = next_model_index
                        next_model_index += 1

                    if next_model_index == 0:
                        raise ValueError(
                            "No devices are enabled for model loading.")
                    self._device_number_to_index = device_number_to_index
            else:
                self._device_number_to_index = []

            self._enable_devices_numbers = None

        return self._device_number_to_index


class DeepSeekOCRHuggingFaceModel(HuggingFaceBackend):
    def __init__(
        self,
        model_path: Path | None,
        local_only: bool,
        enable_devices_numbers: Iterable[int] | None,
    ) -> None:
        super().__init__(
            model_name="deepseek-ai/DeepSeek-OCR",
            model_path=model_path,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
            default_revision=_DEEPSEEK_OCR_REVISION,
        )


class DeepSeekOCR2HuggingFaceModel(HuggingFaceBackend):
    def __init__(
        self,
        model_path: Path | None,
        local_only: bool,
        enable_devices_numbers: Iterable[int] | None,
    ) -> None:
        super().__init__(
            model_name="deepseek-ai/DeepSeek-OCR-2",
            model_path=model_path,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
            default_revision=_DEEPSEEK_OCR2_REVISION,
            attn_implementation="flash_attention_2" if find_spec("flash_attn") is not None else "eager",
        )


class UnlimitedOCRHuggingFaceModel(HuggingFaceBackend):
    def __init__(
        self,
        model_path: Path | None,
        local_only: bool,
        enable_devices_numbers: Iterable[int] | None,
    ) -> None:
        super().__init__(
            model_name="baidu/Unlimited-OCR",
            model_path=model_path,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
            default_revision=_UNLIMITED_OCR_REVISION,
        )

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> str:
        if size not in _UNLIMITED_SIZE_CONFIGS:
            raise ValueError("Unlimited OCR local supports only base and gundam sizes.")

        tokenizer, llm_model = self._select_model(device_number)
        config = _UNLIMITED_SIZE_CONFIGS[size]

        with self._rwlock.gen_rlock():
            with InferWithInterruption(llm_model, context) as infer:
                text_result = infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(output_path),
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    save_results=False,
                    eval_mode=True,
                    max_length=32768,
                    no_repeat_ngram_size=35,
                    ngram_window=128,
                    temperature=0.0,
                )

        output_path.mkdir(parents=True, exist_ok=True)
        raw_text = str(text_result[0] if isinstance(text_result, tuple) else text_result)
        (output_path / "result.md").write_text(raw_text, encoding="utf-8")
        return raw_text
