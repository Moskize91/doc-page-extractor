import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Protocol, cast

from ..parser import ParsedItemKind, parse_ocr_response
from ..structure import build_structured_page, deepseek_ref_to_kind
from ..types import DeepSeekOCRSize, ExtractionContext, Layout, LayoutKind, OCRPageResult

_DEFAULT_VENDOR_MAX_TOKENS = 8000
_LINE_BLOCK_PATTERN = re.compile(
    r"^(?P<ref>[A-Za-z_]+)\[\[(?P<x1>\d+),\s*(?P<y1>\d+),\s*(?P<x2>\d+),\s*(?P<y2>\d+)\]\]\s*$",
    re.MULTILINE,
)

if TYPE_CHECKING:
    from PIL import Image
    import requests


class _ImageLike(Protocol):
    size: tuple[int, int]


DeepSeekLayoutParser = Callable[[_ImageLike, str, str], list[Layout]]


class _LocalDeepSeekModel(Protocol):
    def download(self, revision: str | None) -> None:
        ...

    def load(self) -> None:
        ...

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> str:
        ...


def parse_deepseek_ocr_layouts(
    image: _ImageLike, response: str, source: str = "deepseek-ocr-vendor"
) -> list[Layout]:
    return [
        _deepseek_layout(label=label, det=det, text=text, source=source)
        for label, det, text in _parse_deepseek_ocr_response(image, response)
        if _has_area(det)
    ]


def parse_deepseek_ocr2_layouts(
    image: _ImageLike, response: str, source: str = "deepseek-ocr2-vendor"
) -> list[Layout]:
    return [
        _deepseek_layout(label=label, det=det, text=text, source=source)
        for label, det, text in _parse_deepseek_ocr2_response(image, response)
        if _has_area(det)
    ]


def _deepseek_layout(
    label: str,
    det: tuple[int, int, int, int],
    text: str | None,
    source: str,
) -> Layout:
    kind = deepseek_ref_to_kind(label, text)
    return Layout(
        det=det,
        text=text,
        type=label,
        html=text if kind == LayoutKind.TABLE and _looks_like_html_table(text) else None,
        kind=kind,
        source=source,
    )


class _DeepSeekLocalAdapter:
    allows_multi_stage = True

    def __init__(
        self,
        model: _LocalDeepSeekModel,
        source: str = "deepseek-ocr",
        parse_layouts: DeepSeekLayoutParser = parse_deepseek_ocr_layouts,
    ) -> None:
        self._model = model
        self._source = source
        self._parse_layouts = parse_layouts

    def download(self, revision: str | None) -> None:
        self._model.download(revision)

    def load(self) -> None:
        self._model.load()

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        response = self._model.generate(
            prompt=prompt,
            image_path=image_path,
            output_path=output_path,
            size=size,
            context=context,
            device_number=device_number,
        )
        from PIL import Image

        with Image.open(image_path) as image:
            layouts = self._parse_layouts(image, response, self._source)
        return OCRPageResult(
            layouts=layouts,
            source=self._source,
            structured=build_structured_page(layouts),
            raw_text=response,
        )


@dataclass
class DeepSeekOCRVendorConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = _DEFAULT_VENDOR_MAX_TOKENS
    timeout_seconds: int = 180


@dataclass
class DeepSeekOCR2VendorConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = _DEFAULT_VENDOR_MAX_TOKENS
    timeout_seconds: int = 180


class DeepSeekOCRVendorAdapter:
    allows_multi_stage = True

    def __init__(self, config: DeepSeekOCRVendorConfig) -> None:
        self._config = config

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        del output_path, size, device_number
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_path)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p

        import requests

        response = requests.post(
            _vendor_chat_completions_url(self._config.base_url),
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "doc-page-extractor-deepseek-ocr-vendor/1.0",
            },
            json=payload,
            timeout=self._config.timeout_seconds,
        )
        if response.status_code >= 400:
            _raise_vendor_error(response)

        data = response.json()
        usage = data.get("usage") or {}
        if context is not None:
            context.input_tokens += int(usage.get("prompt_tokens") or 0)
            context.output_tokens += int(usage.get("completion_tokens") or 0)

        choices = data.get("choices") or []
        raw_text = ""
        if choices:
            raw_text = str((choices[0].get("message") or {}).get("content") or "")

        from PIL import Image

        with Image.open(image_path) as image:
            layouts = parse_deepseek_ocr_layouts(
                image, raw_text, source="deepseek-ocr-vendor"
            )
        return OCRPageResult(
            layouts=layouts,
            source="deepseek-ocr-vendor",
            structured=build_structured_page(layouts),
            raw_text=raw_text,
            raw={"usage": usage},
        )


class DeepSeekOCR2VendorAdapter:
    allows_multi_stage = True

    def __init__(self, config: DeepSeekOCR2VendorConfig) -> None:
        self._config = config

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        del output_path, size, device_number
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_path)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p

        import requests

        response = requests.post(
            _vendor_chat_completions_url(self._config.base_url),
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "doc-page-extractor-deepseek-ocr2-vendor/1.0",
            },
            json=payload,
            timeout=self._config.timeout_seconds,
        )
        if response.status_code >= 400:
            _raise_vendor_error(response)

        data = response.json()
        usage = data.get("usage") or {}
        if context is not None:
            context.input_tokens += int(usage.get("prompt_tokens") or 0)
            context.output_tokens += int(usage.get("completion_tokens") or 0)

        choices = data.get("choices") or []
        raw_text = ""
        if choices:
            raw_text = str((choices[0].get("message") or {}).get("content") or "")

        from PIL import Image

        with Image.open(image_path) as image:
            layouts = parse_deepseek_ocr2_layouts(image, raw_text)
        return OCRPageResult(
            layouts=layouts,
            source="deepseek-ocr2-vendor",
            structured=build_structured_page(layouts),
            raw_text=raw_text,
            raw={"usage": usage},
        )


def _parse_deepseek_ocr_response(
    image: _ImageLike, response: str
) -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
    width, height = image.size
    det: tuple[int, int, int, int] | None = None
    ref: str | None = None

    for kind, content in parse_ocr_response(response, width, height):
        if kind == ParsedItemKind.TEXT:
            if det is not None and ref is not None:
                yield ref, det, cast(str, content)
                det = None
                ref = None
        if det is not None and ref is not None:
            yield ref, det, None
            det = None
            ref = None
        elif kind == ParsedItemKind.DET:
            det = cast(tuple[int, int, int, int], content)
        elif kind == ParsedItemKind.REF:
            ref = cast(str, content)
    if det is not None and ref is not None:
        yield ref, det, None


def _parse_deepseek_ocr2_response(
    image: _ImageLike, response: str
) -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
    if not _LINE_BLOCK_PATTERN.search(response):
        yield from _parse_deepseek_ocr_response(image, response)
        return

    yield from _parse_deepseek_ocr2_line_blocks(image, response)


def _parse_deepseek_ocr2_line_blocks(
    image: _ImageLike,
    response: str,
) -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
    width, height = image.size
    ref: str | None = None
    det: tuple[int, int, int, int] | None = None
    text_parts: list[str] = []

    def flush() -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
        nonlocal ref, det, text_parts
        if ref is not None and det is not None:
            text = "\n".join(text_parts).strip("\n") or None
            yield ref, det, text
        ref = None
        det = None
        text_parts = []

    for line in response.splitlines():
        matched = _LINE_BLOCK_PATTERN.match(line)
        if matched:
            yield from flush()
            ref = matched.group("ref")
            x1_norm = int(matched.group("x1"))
            y1_norm = int(matched.group("y1"))
            x2_norm = int(matched.group("x2"))
            y2_norm = int(matched.group("y2"))
            det = (
                round(x1_norm / 1000 * width),
                round(y1_norm / 1000 * height),
                round(x2_norm / 1000 * width),
                round(y2_norm / 1000 * height),
            )
            continue
        if ref is not None and det is not None:
            text_parts.append(line)

    yield from flush()


def _has_area(det: tuple[int, int, int, int]) -> bool:
    return det[2] > det[0] and det[3] > det[1]


def _looks_like_html_table(text: str | None) -> bool:
    return bool(text and text.lstrip().lower().startswith("<table"))


def _vendor_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _raise_vendor_error(response: Any) -> None:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    raise RuntimeError(
        f"DeepSeek Vendor request failed with HTTP {response.status_code}: "
        f"{body[:500]}"
    )
