"""Adapter for the GLM-OCR SDK's structured parsing service."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..extraction_context import AbortError, TokenLimitError
from ..structure import build_structured_page, unlimited_ocr_type_to_kind
from ..types import DeepSeekOCRSize, ExtractionContext, Layout, LayoutKind, OCRPageResult

_DEFAULT_ENDPOINT_URL = "http://127.0.0.1:5002/glmocr/parse"
_SOURCE = "glm-ocr-sdk"
_MISSING = object()


@dataclass
class GLMOCRServiceConfig:
    """Client settings for an externally managed GLM-OCR SDK service.

    The endpoint must be the SDK's full ``/glmocr/parse`` endpoint, not the
    underlying OpenAI-compatible MLX/vLLM endpoint.  The SDK server owns model
    selection, layout configuration, and token generation settings.
    """

    endpoint_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 180

    def __post_init__(self) -> None:
        if self.endpoint_url is None:
            self.endpoint_url = _DEFAULT_ENDPOINT_URL
        elif not isinstance(self.endpoint_url, str) or not self.endpoint_url.strip():
            raise ValueError("endpoint_url must be a non-empty string")
        else:
            self.endpoint_url = self.endpoint_url.strip()
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        try:
            numeric_timeout = float(self.timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "timeout_seconds must be a finite positive number"
            ) from error
        if not math.isfinite(numeric_timeout) or numeric_timeout <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")


class GLMOCRServiceAdapter:
    """Call the full GLM-OCR SDK parsing service over HTTP.

    The SDK returns normalized 0-1000 coordinates and the final formatter's
    reading-order ``index``.  This adapter converts those coordinates to the
    exact pixel dimensions of the uploaded image and retains the SDK's native
    label in ``Layout.type`` and ``Layout.raw``.

    The service does not expose a cancellation endpoint.  Aborts are checked
    before and after the HTTP request; an in-flight request can only be
    bounded by ``timeout_seconds``.  The SDK server currently returns an empty
    usage object, so ExtractionContext token limits are rejected rather than
    guessed and token counters are updated only for explicit usage fields.
    """

    allows_multi_stage = False

    def __init__(self, config: GLMOCRServiceConfig) -> None:
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
        del prompt, output_path, size, device_number
        _check_aborted(context)
        _validate_token_limits(context)

        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(image_path) as image:
            image_size = image.size
        payload = {"images": [_image_data_url(image_path)]}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "doc-page-extractor-glm-ocr-sdk/1.0",
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        import requests  # type: ignore[import-not-found]

        endpoint_url = self._config.endpoint_url
        if endpoint_url is None:  # defensive: __post_init__ supplies the default
            raise RuntimeError("GLM-OCR endpoint URL is not configured.")
        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=self._config.timeout_seconds,
            )
        except requests.exceptions.Timeout as error:
            raise TimeoutError(
                "GLM-OCR SDK service request timed out after "
                f"{self._config.timeout_seconds} seconds."
            ) from error

        if response.status_code >= 400:
            raise RuntimeError(
                "GLM-OCR SDK service request failed with HTTP "
                f"{response.status_code}: {_response_preview(response)}"
            )

        try:
            response_data = response.json()
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                "GLM-OCR SDK service returned invalid JSON."
            ) from error
        if not isinstance(response_data, dict):
            raise RuntimeError(
                "GLM-OCR SDK service returned a non-object JSON response."
            )
        if response_data.get("error"):
            raise RuntimeError(
                f"GLM-OCR SDK service returned an error: "
                f"{_json_preview(response_data['error'])}"
            )

        _update_usage(context, response_data.get("usage"))
        _check_aborted(context)

        markdown_result = response_data.get("markdown_result")
        if not markdown_result:
            markdown_result = response_data.get("md_results") or ""
        if not isinstance(markdown_result, str):
            markdown_result = str(markdown_result)

        layouts = parse_glm_ocr_layouts(
            response_data,
            image_size=image_size,
            source=_SOURCE,
            markdown_result=markdown_result,
        )
        return OCRPageResult(
            layouts=layouts,
            source=_SOURCE,
            raw_text=markdown_result,
            raw=response_data,
            structured=build_structured_page(layouts),
        )


def parse_glm_ocr_layouts(
    response: Mapping[str, Any],
    image_size: tuple[int, int],
    source: str = _SOURCE,
    markdown_result: str = "",
) -> list[Layout]:
    """Map one SDK response into pixel-coordinate layouts.

    The full SDK service returns ``json_result`` as a list containing one page
    (it aliases this as ``layout_details``).  A direct region list is accepted
    too because it is documented by older SDK response examples.  More than
    one page is rejected: this adapter uploads exactly one page and must not
    silently merge page coordinates.
    """

    if not markdown_result:
        for key in ("markdown_result", "md_results"):
            candidate = response.get(key)
            if isinstance(candidate, str) and candidate:
                markdown_result = candidate
                break
    regions = _response_regions(response, markdown_result)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid input image dimensions: {image_size!r}")

    ordered_regions: list[tuple[int, int, Mapping[str, Any]]] = []
    for response_position, region in enumerate(regions):
        if not isinstance(region, Mapping):
            raise ValueError(
                "GLM-OCR SDK response contains a non-object layout region."
            )
        index = region.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                "GLM-OCR SDK response region has no valid non-negative index."
            )
        ordered_regions.append((index, response_position, region))
    ordered_regions.sort(key=lambda item: (item[0], item[1]))

    layouts: list[Layout] = []
    seen_indices: set[int] = set()
    for index, _response_position, region in ordered_regions:
        if index in seen_indices:
            raise ValueError(
                f"GLM-OCR SDK response contains duplicate region index {index}."
            )
        seen_indices.add(index)

        text = _optional_text(region.get("content"))
        native_label = _native_label(region)
        kind = _glm_label_to_kind(native_label, region.get("label"), text)
        # The SDK adds Markdown heading markers to structured title content.
        # Downstream renderers already know the block kind; do not duplicate them.
        if kind == LayoutKind.TITLE and text is not None:
            text = re.sub(r"^#{1,6}\s+", "", text)
        det = _pixel_box(region.get("bbox_2d"), width, height)
        polygon = _pixel_polygon(region.get("polygon"), width, height)
        layouts.append(
            Layout(
                det=det,
                text=text,
                type=native_label,
                polygon=polygon,
                html=_table_html(text, kind),
                source=source,
                raw=dict(region),
                kind=kind,
            )
        )
    return layouts


def _response_regions(
    response: Mapping[str, Any], markdown_result: str
) -> list[Mapping[str, Any]]:
    value: Any = response.get("json_result", _MISSING)
    if value is _MISSING or value is None:
        value = response.get("layout_details", _MISSING)

    if value is _MISSING or value is None:
        raise ValueError(
            "GLM-OCR SDK response is missing structured layout geometry."
        )
    if not isinstance(value, list):
        raise ValueError("GLM-OCR SDK layout result must be a list.")
    if not value:
        if markdown_result.strip():
            raise ValueError(
                "GLM-OCR SDK returned Markdown without structured layout geometry."
            )
        return []

    if all(isinstance(item, Mapping) for item in value):
        return value
    if not all(isinstance(item, list) for item in value):
        raise ValueError("GLM-OCR SDK layout result has an invalid page shape.")
    if len(value) != 1:
        raise ValueError(
            "GLM-OCR SDK returned multiple pages for a single uploaded image."
        )
    page = value[0]
    if not all(isinstance(item, Mapping) for item in page):
        raise ValueError("GLM-OCR SDK page layout result has an invalid region.")
    if not page and markdown_result.strip():
        raise ValueError(
            "GLM-OCR SDK returned Markdown without structured layout geometry."
        )
    return page


def _native_label(region: Mapping[str, Any]) -> str:
    value = region.get("native_label")
    if value is None:
        value = region.get("label")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GLM-OCR SDK response region has no usable label.")
    return value.strip()


def _glm_label_to_kind(
    native_label: str, output_label: Any, text: str | None
) -> LayoutKind:
    native = native_label.strip().lower()
    output = output_label.strip().lower() if isinstance(output_label, str) else ""

    explicit = {
        "doc_title": LayoutKind.TITLE,
        "paragraph_title": LayoutKind.TITLE,
        "title": LayoutKind.TITLE,
        "display_formula": LayoutKind.EQUATION,
        "inline_formula": LayoutKind.EQUATION,
        "formula": LayoutKind.EQUATION,
        "chart": LayoutKind.IMAGE,
        "image": LayoutKind.IMAGE,
        "vision_footnote": LayoutKind.FOOTNOTE,
        "footnote": LayoutKind.FOOTNOTE,
        "reference": LayoutKind.TEXT,
        "seal": LayoutKind.TEXT,
        "formula_number": LayoutKind.TEXT,
        "abstract": LayoutKind.TEXT,
        "algorithm": LayoutKind.TEXT,
        "content": LayoutKind.TEXT,
        "reference_content": LayoutKind.TEXT,
        "vertical_text": LayoutKind.TEXT,
        "header_image": LayoutKind.HEADER,
        "footer_image": LayoutKind.FOOTER,
    }
    if native in explicit:
        return explicit[native]

    native_kind = unlimited_ocr_type_to_kind(native, text)
    if native_kind != LayoutKind.UNKNOWN:
        return native_kind
    output_kind = unlimited_ocr_type_to_kind(output, text)
    if output_kind != LayoutKind.UNKNOWN:
        return output_kind
    raise ValueError(
        f"GLM-OCR SDK returned an unsupported layout label: {native_label!r}."
    )


def _pixel_box(
    value: Any, width: int, height: int
) -> tuple[int, int, int, int]:
    normalized = _normalized_numbers(value, 4, "bbox_2d")
    x1, y1, x2, y2 = (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
    )
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"GLM-OCR SDK returned a degenerate bbox_2d: {value!r}")
    det = (
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    )
    if det[0] >= det[2] or det[1] >= det[3]:
        raise ValueError(
            f"GLM-OCR SDK bbox_2d collapses at input pixel size: {value!r}"
        )
    return det


def _pixel_polygon(
    value: Any, width: int, height: int
) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError("GLM-OCR SDK returned an invalid polygon.")
    points: list[tuple[int, int]] = []
    for point in value:
        normalized = _normalized_numbers(point, 2, "polygon point")
        points.append(
            (
                round(normalized[0] * width / 1000),
                round(normalized[1] * height / 1000),
            )
        )
    if len(set(points)) < 3:
        raise ValueError("GLM-OCR SDK returned a degenerate polygon.")
    return points


def _normalized_numbers(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"GLM-OCR SDK returned an invalid {name}: {value!r}")
    numbers: list[float] = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, (int, float)):
            raise ValueError(f"GLM-OCR SDK returned an invalid {name}: {value!r}")
        try:
            number = float(part)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"GLM-OCR SDK returned an invalid {name}: {value!r}") from error
        if not math.isfinite(number) or not 0 <= number <= 1000:
            raise ValueError(
                f"GLM-OCR SDK returned out-of-range normalized {name}: {value!r}"
            )
        numbers.append(number)
    return numbers


def _table_html(text: str | None, kind: LayoutKind) -> str | None:
    if kind != LayoutKind.TABLE or text is None:
        return None
    stripped = text.strip()
    if stripped.lower().startswith("<table") and stripped.lower().endswith(
        "</table>"
    ):
        return stripped
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{encoded}"


def _check_aborted(context: ExtractionContext | None) -> None:
    if context is None:
        return
    if context.check_aborted():
        error = AbortError()
        error.input_tokens = context.input_tokens
        error.output_tokens = context.output_tokens
        raise error


def _validate_token_limits(context: ExtractionContext | None) -> None:
    if context is None:
        return
    total_limit = context.max_tokens
    if total_limit is not None and total_limit <= (
        context.input_tokens + context.output_tokens
    ):
        error = TokenLimitError()
        error.input_tokens = context.input_tokens
        error.output_tokens = context.output_tokens
        raise error
    output_limit = context.max_output_tokens
    if output_limit is not None and output_limit <= context.output_tokens:
        error = TokenLimitError()
        error.input_tokens = context.input_tokens
        error.output_tokens = context.output_tokens
        raise error
    if total_limit is not None or output_limit is not None:
        raise NotImplementedError(
            "GLM-OCR SDK service does not expose token limits or complete usage; "
            "ExtractionContext token limits are unsupported."
        )


def _update_usage(context: ExtractionContext | None, usage: Any) -> None:
    if context is None or not isinstance(usage, Mapping):
        return
    for key, attribute in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
    ):
        value = usage.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"GLM-OCR SDK returned invalid usage.{key}.")
        try:
            numeric_value = float(value)
            converted_value = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"GLM-OCR SDK returned invalid usage.{key}.") from error
        if (
            not math.isfinite(numeric_value)
            or numeric_value < 0
            or converted_value != numeric_value
        ):
            raise RuntimeError(f"GLM-OCR SDK returned invalid usage.{key}.")
        if attribute == "input_tokens":
            context.input_tokens += converted_value
        else:
            context.output_tokens += converted_value


def _response_preview(response: Any) -> str:
    try:
        body = response.json()
    except (ValueError, TypeError, AttributeError):
        body = getattr(response, "text", "")
    return _json_preview(body)


def _json_preview(value: Any) -> str:
    if isinstance(value, str):
        return value[:500]
    try:
        return json.dumps(value, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return str(value)[:500]


__all__ = [
    "GLMOCRServiceAdapter",
    "GLMOCRServiceConfig",
    "parse_glm_ocr_layouts",
]
