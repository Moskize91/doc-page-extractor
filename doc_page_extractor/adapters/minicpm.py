"""MiniCPM-OCR document parsing, independent of its HTTP transport."""

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import VendorOCRRequestError
from ..structure import build_structured_page
from ..types import DeepSeekOCRSize, ExtractionContext, Layout, LayoutKind, OCRPageResult

_SOURCE = "minicpm-ocr-vendor"
_CAPTIONS = {
    LayoutKind.IMAGE: LayoutKind.IMAGE_CAPTION,
    LayoutKind.TABLE: LayoutKind.TABLE_CAPTION,
    LayoutKind.EQUATION: LayoutKind.EQUATION_CAPTION,
}
_LABELS = {
    **dict.fromkeys(("doc_title", "paragraph_title", "sub_title", "title"), LayoutKind.TITLE),
    **dict.fromkeys(("figure", "image"), LayoutKind.IMAGE),
    **dict.fromkeys(("chart", "table"), LayoutKind.TABLE),
    **dict.fromkeys(("display_formula", "equation", "formula", "inline_formula"),
                    LayoutKind.EQUATION),
    **dict.fromkeys(("header", "header_image"), LayoutKind.HEADER),
    **dict.fromkeys(("footer", "footer_image"), LayoutKind.FOOTER),
    **dict.fromkeys(("aside", "aside_text"), LayoutKind.ASIDE),
    "number": LayoutKind.PAGE_NUMBER,
    "footnote": LayoutKind.FOOTNOTE,
}
_CAPTION_LABELS = {
    "figure_title": {LayoutKind.IMAGE, LayoutKind.TABLE},
    "vision_footnote": {LayoutKind.IMAGE, LayoutKind.TABLE},
    "image_caption": {LayoutKind.IMAGE},
    "table_caption": {LayoutKind.TABLE},
    "equation_caption": {LayoutKind.EQUATION},
    "formula_number": {LayoutKind.EQUATION},
}


def parse_minicpm_ocr_layouts(
    data: Mapping[str, Any],
    image_size: tuple[int, int],
    source: str = _SOURCE,
) -> list[Layout]:
    """Decode hosted ``ocr`` or self-hosted ``minicpm_ocr`` pixel regions.

    Explicit recognition failures reject the page. Captions are placed next to
    their selected asset so the common structured-page builder keeps ownership.
    """
    page = next((data[key] for key in ("ocr", "minicpm_ocr")
                 if isinstance(data.get(key), Mapping)), None)
    if page is None:
        raise ValueError("MiniCPM-OCR response requires an 'ocr' or 'minicpm_ocr' page")
    for payload in (data, page):
        rotation = payload.get("rotation")
        applied = rotation.get("applied") if isinstance(rotation, Mapping) else False
        if _is_true(applied) or _is_true(payload.get("rotated")):
            raise ValueError("MiniCPM-OCR orientation correction must be disabled")
    blocks = page.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("MiniCPM-OCR requires document layout blocks, not a plain vision endpoint")
    width, height = image_size
    reported = (page.get("image_width"), page.get("image_height"))
    if reported == (None, None):
        scale_x = scale_y = 1.0
    else:
        if not all(_positive_number(value) for value in reported):
            raise ValueError("MiniCPM-OCR requires positive image_width and image_height")
        scale_x, scale_y = width / reported[0], height / reported[1]

    def order(item):
        index, block = item
        value = block.get("order") if isinstance(block, Mapping) else None
        return (value if isinstance(value, int) and not isinstance(value, bool) else index, index)

    layouts = []
    for _, block in sorted(enumerate(blocks), key=order):
        if not isinstance(block, Mapping):
            raise ValueError("MiniCPM-OCR returned an invalid block")
        status = block.get("status")
        if status not in (None, "ok", "skipped_no_vlm", "skipped_seal"):
            raise ValueError(f"MiniCPM-OCR block {block.get('order')} failed: {status}")
        label = str(block.get("label") or "").strip().lower()
        content = block.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("MiniCPM-OCR block content must be text")
        kind = _LABELS.get(label, LayoutKind.TEXT)
        if status == "skipped_seal" or (kind == LayoutKind.TABLE and not content.strip()):
            kind = LayoutKind.IMAGE
        if not content.strip() and kind != LayoutKind.IMAGE:
            continue
        det = _box(block.get("bbox"))
        if det is None:
            regions = block.get("regions") or []
            boxes = [_box(region) for region in regions]
            if not boxes or any(box is None for box in boxes):
                raise ValueError("MiniCPM-OCR block has no usable region")
            det = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                   max(box[2] for box in boxes), max(box[3] for box in boxes))
        det = (max(0, min(width, round(det[0] * scale_x))),
               max(0, min(height, round(det[1] * scale_y))),
               max(0, min(width, round(det[2] * scale_x))),
               max(0, min(height, round(det[3] * scale_y))))
        if det[0] >= det[2] or det[1] >= det[3]:
            raise ValueError("MiniCPM-OCR block has no usable region")
        layouts.append(Layout(
            det=det, text=content or None, type=label,
            html=content if kind == LayoutKind.TABLE else None,
            kind=kind, source=source, raw=dict(block),
        ))
    if blocks and not layouts:
        raise ValueError("MiniCPM-OCR returned no usable regions")
    return _attach_captions(layouts)


def _is_true(value: Any) -> bool:
    return value is True or value == 1 or (isinstance(value, str) and value.lower() in {"true", "1"})


def _positive_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        return None
    return tuple(value) if value[0] < value[2] and value[1] < value[3] else None


def _attach_captions(layouts: list[Layout]) -> list[Layout]:
    attachments: dict[int, list[Layout]] = {}
    attached: set[int] = set()
    for index, caption in enumerate(layouts):
        allowed = _CAPTION_LABELS.get(caption.type or "")
        if allowed is None:
            continue
        candidates = []
        for direction in (-1, 1):
            neighbor = index + direction
            while 0 <= neighbor < len(layouts):
                asset = layouts[neighbor]
                if asset.type in _CAPTION_LABELS:
                    neighbor += direction
                    continue
                same_column = min(caption.det[2], asset.det[2]) > max(caption.det[0], asset.det[0])
                same_row = (caption.type == "formula_number" and
                            min(caption.det[3], asset.det[3]) > max(caption.det[1], asset.det[1]))
                if asset.kind in allowed and (same_column or same_row):
                    gap = max(0, asset.det[1] - caption.det[3], caption.det[1] - asset.det[3])
                    candidates.append((gap, neighbor))
                break
        if not candidates:
            continue  # Preserve unattached captions and equation numbers as text.
        _, owner = min(candidates)
        caption.kind = _CAPTIONS[layouts[owner].kind]
        attachments.setdefault(owner, []).append(caption)
        attached.add(index)
    result = []
    for index, layout in enumerate(layouts):
        if index not in attached:
            result.append(layout)
            result.extend(attachments.get(index, []))
    return result


@dataclass(frozen=True)
class MiniCPMOCRVendorConfig:
    base_url: str
    api_key: str = field(default="", repr=False)
    model: str = "MiniCPM-OCR-1.0"
    temperature: float | None = 0.0
    top_p: float | None = None
    max_tokens: int = 8000
    timeout_seconds: int = 180


class MiniCPMOCRVendorAdapter:
    """Single-stage adapter for a full MiniCPM-OCR document-parsing service."""

    allows_multi_stage = False
    prompt = "Text Recognition:"

    def __init__(self, config: MiniCPMOCRVendorConfig) -> None:
        self._config = config

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def extract_page(
        self, prompt: str, image_path: Path, output_path: Path,
        size: DeepSeekOCRSize, context: ExtractionContext | None, device_number: int | None,
    ) -> OCRPageResult:
        del output_path, size, device_number
        from PIL import Image
        import requests

        config = self._config
        image_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]}],
            "minicpm_ocr": {"mode": "ocr_only", "upright": False},
            "max_tokens": config.max_tokens, "stream": False,
        }
        for key in ("temperature", "top_p"):
            if getattr(config, key) is not None:
                payload[key] = getattr(config, key)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        url = config.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions" if url.endswith("/v1") else "/v1/chat/completions"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as error:
            raise VendorOCRRequestError("MiniCPM-OCR Vendor request failed") from error
        try:
            data = response.json()
            with Image.open(image_path) as image:
                layouts = parse_minicpm_ocr_layouts(data, image.size)
            usage = data.get("usage") or {}
            if context is not None:
                context.input_tokens += int(usage.get("prompt_tokens") or 0)
                context.output_tokens += int(usage.get("completion_tokens") or 0)
        except (ValueError, TypeError, AttributeError) as error:
            raise VendorOCRRequestError("MiniCPM-OCR Vendor returned an invalid response") from error
        choices = data.get("choices") or []
        raw_text = (choices[0].get("message") or {}).get("content") if choices else None
        return OCRPageResult(layouts=layouts, source=_SOURCE,
                             structured=build_structured_page(layouts), raw_text=raw_text, raw=data)
