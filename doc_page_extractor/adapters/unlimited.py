import base64
import ast
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol

from ..structure import unlimited_ocr_type_to_kind, build_structured_page
from ..types import DeepSeekOCRSize, ExtractionContext, Layout, OCRPageResult

if TYPE_CHECKING:
    import requests

_LOCAL_PROMPT = "<image>document parsing."
_LOCAL_DET_PATTERN = re.compile(
    r"<\|det\|>\s*"
    r"(?P<type>[A-Za-z_][\w-]*)"
    r"\s*(?P<coords>\[[\s\S]*?\])?"
    r"\s*<\|/det\|>",
)


class _LocalUnlimitedModel(Protocol):
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


@dataclass
class UnlimitedOCRVendorConfig:
    ak: str
    sk: str
    base_url: str = "https://aip.baidubce.com"
    poll_interval_seconds: float = 2
    timeout_seconds: int = 180


class UnlimitedOCRVendorAdapter:
    allows_multi_stage = False
    max_image_side = 8192

    def __init__(self, config: UnlimitedOCRVendorConfig) -> None:
        self._config = config
        self._access_token: str | None = None

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
        token = self._get_access_token()
        task_id = self._submit_task(token, image_path)
        task_result = self._wait_for_task(token, task_id, context)
        parse_url = str(task_result.get("parse_result_url") or "")
        if not parse_url:
            raise RuntimeError(
                f"Unlimited OCR task {task_id} did not return parse_result_url."
            )

        parse_result = self._download_parse_result(parse_url)
        layouts = parse_unlimited_ocr_layouts(
            parse_result,
            source="unlimited-ocr-vendor",
        )
        return OCRPageResult(
            layouts=layouts,
            source="unlimited-ocr-vendor",
            structured=build_structured_page(layouts),
            raw={
                "task_id": task_id,
                "status": task_result.get("status"),
                "file_name": parse_result.get("file_name"),
            },
        )

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        import requests

        response = requests.post(
            f"{self._config.base_url.rstrip('/')}/oauth/2.0/token",
            headers={
                "Accept": "application/json",
                "User-Agent": "doc-page-extractor-unlimited-ocr/1.0",
            },
            data={
                "grant_type": "client_credentials",
                "client_id": self._config.ak,
                "client_secret": self._config.sk,
            },
            timeout=self._config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Unlimited OCR token request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        token = str(data.get("access_token") or "")
        if not token:
            raise RuntimeError(
                f"Unlimited OCR token response did not include access_token: {data}"
            )
        self._access_token = token
        return token

    def _submit_task(self, token: str, image_path: Path) -> str:
        import requests

        response = requests.post(
            self._api_url("/rest/2.0/brain/online/v2/unlimited-ocr-parser/task", token),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "doc-page-extractor-unlimited-ocr/1.0",
            },
            data={
                "file_data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "file_name": image_path.name,
            },
            timeout=self._config.timeout_seconds,
        )
        data = self._checked_response(response, "submit")
        task_id = str((data.get("result") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError(
                f"Unlimited OCR submit response did not include task_id: {data}"
            )
        return task_id

    def _wait_for_task(
        self, token: str, task_id: str, context: ExtractionContext | None
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._config.timeout_seconds
        while True:
            if context is not None:
                context.check_aborted()
            import requests

            response = requests.post(
                self._api_url(
                    "/rest/2.0/brain/online/v2/unlimited-ocr-parser/task/query",
                    token,
                ),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": "doc-page-extractor-unlimited-ocr/1.0",
                },
                data={"task_id": task_id},
                timeout=self._config.timeout_seconds,
            )
            data = self._checked_response(response, "query")
            result = data.get("result") or {}
            status = result.get("status")
            if status == "success" or result.get("parse_result_url"):
                return result
            if status == "failed":
                raise RuntimeError(f"Unlimited OCR task {task_id} failed: {result}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Unlimited OCR task {task_id} timed out.")
            time.sleep(self._config.poll_interval_seconds)

    def _download_parse_result(self, url: str) -> dict[str, Any]:
        import requests

        response = requests.get(
            url,
            headers={"User-Agent": "doc-page-extractor-unlimited-ocr/1.0"},
            timeout=self._config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Unlimited OCR parse result download failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        return json.loads(response.content.decode("utf-8"))

    def _api_url(self, path: str, token: str) -> str:
        query = urllib.parse.urlencode({"access_token": token})
        return f"{self._config.base_url.rstrip('/')}{path}?{query}"

    @staticmethod
    def _checked_response(response: Any, action: str) -> dict[str, Any]:
        if response.status_code >= 400:
            raise RuntimeError(
                f"Unlimited OCR {action} request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        if int(data.get("error_code") or 0) != 0:
            raise RuntimeError(f"Unlimited OCR {action} request failed: {data}")
        return data


class UnlimitedModelOCRAdapter:
    allows_multi_stage = False
    prompt = _LOCAL_PROMPT

    def __init__(self, model: _LocalUnlimitedModel, source: str = "unlimited-ocr") -> None:
        self._model = model
        self._source = source

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
            layouts = parse_unlimited_ocr_local_layouts(
                image,
                response,
                source=self._source,
            )
        return OCRPageResult(
            layouts=layouts,
            source=self._source,
            structured=build_structured_page(layouts),
            raw_text=response,
        )


def parse_unlimited_ocr_layouts(
    parse_result: dict[str, Any],
    source: str = "unlimited-ocr-vendor",
) -> list[Layout]:
    layouts: list[Layout] = []
    for page in parse_result.get("pages") or []:
        for item in page.get("layouts") or []:
            det = _position_to_det(item.get("position"))
            if det is None:
                det = _polygon_to_det(item.get("polygon"))
            if det is None:
                continue
            layout_type = _optional_str(item.get("type"))
            text = _optional_str(item.get("text"))
            table_html = _optional_str(item.get("table_html"))
            html = table_html or (text if layout_type == "table" else None)
            kind = unlimited_ocr_type_to_kind(layout_type, text)
            layouts.append(
                Layout(
                    det=det,
                    text=text,
                    kind=kind,
                    type=layout_type,
                    polygon=_parse_polygon(item.get("polygon")),
                    html=html,
                    source=source,
                    raw=item if isinstance(item, dict) else None,
                )
            )
    return layouts


def parse_unlimited_ocr_local_layouts(
    image: Any,
    response: str,
    source: str = "unlimited-ocr",
) -> list[Layout]:
    width, height = image.size
    layouts: list[Layout] = []
    matches = list(_LOCAL_DET_PATTERN.finditer(response))

    for index, matched in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        layout_type = matched.group("type")
        text = response[matched.end():next_start].strip("\n") or None
        for det in _parse_local_dets(matched.group("coords"), width, height):
            kind = unlimited_ocr_type_to_kind(layout_type, text)
            layouts.append(
                Layout(
                    det=det,
                    text=text,
                    kind=kind,
                    type=layout_type,
                    html=text if kind.value == "table" and _looks_like_html_table(text) else None,
                    source=source,
                    raw={
                        "type": layout_type,
                        "coords": matched.group("coords"),
                    },
                )
            )
    return layouts


def _parse_local_dets(
    raw_coords: str | None,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    if raw_coords is None:
        return []
    try:
        parsed = ast.literal_eval(raw_coords)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    boxes = parsed if parsed and isinstance(parsed[0], list) else [parsed]
    dets: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not isinstance(box, list) or len(box) < 4:
            continue
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = [float(part) for part in box[:4]]
        except (TypeError, ValueError):
            continue
        det = (
            round(x1_norm / 999 * width),
            round(y1_norm / 999 * height),
            round(x2_norm / 999 * width),
            round(y2_norm / 999 * height),
        )
        if det[2] > det[0] and det[3] > det[1]:
            dets.append(det)
    return dets


def _looks_like_html_table(text: str | None) -> bool:
    return bool(text and text.lstrip().lower().startswith("<table"))


def _position_to_det(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x, y, width, height = [round(float(part)) for part in value[:4]]
    return (x, y, x + width, y + height)


def _polygon_to_det(value: Any) -> tuple[int, int, int, int] | None:
    polygon = _parse_polygon(value)
    if not polygon:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for point in list(polygon):
        xs.append(point[0])
        ys.append(point[1])
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_polygon(value: Any) -> list[tuple[int, int]] | None:
    if not isinstance(value, list):
        return None
    polygon: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, list) or len(point) < 2:
            return None
        polygon.append((round(float(point[0])), round(float(point[1]))))
    return polygon or None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
