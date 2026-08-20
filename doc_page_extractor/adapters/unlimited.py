import base64
import json
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..structure import unlimited_ocr_type_to_kind, build_structured_page, legacy_ref_for_kind
from ..types import DeepSeekOCRSize, ExtractionContext, Layout, OCRPageResult

if TYPE_CHECKING:
    import requests


@dataclass
class UnlimitedOCRConfig:
    ak: str
    sk: str
    base_url: str = "https://aip.baidubce.com"
    poll_interval_seconds: float = 2
    timeout_seconds: int = 180

    @classmethod
    def from_env(cls) -> "UnlimitedOCRConfig":
        import os

        def required(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise SystemExit(f"Missing required environment variable: {name}")
            return value

        def optional_float(name: str, default: float) -> float:
            value = os.environ.get(name, "").strip()
            if not value:
                return default
            try:
                return float(value)
            except ValueError as exc:
                raise SystemExit(f"{name} must be a float, got {value!r}") from exc

        def optional_int(name: str, default: int) -> int:
            value = os.environ.get(name, "").strip()
            if not value:
                return default
            try:
                return int(value)
            except ValueError as exc:
                raise SystemExit(f"{name} must be an integer, got {value!r}") from exc

        return cls(
            ak=required("DPE_UNLIMITED_OCR_ACCESS_KEY"),
            sk=required("DPE_UNLIMITED_OCR_SECRET_KEY"),
            base_url=os.environ.get("DPE_UNLIMITED_OCR_BASE_URL", "https://aip.baidubce.com").strip(),
            poll_interval_seconds=optional_float("DPE_UNLIMITED_OCR_POLL_INTERVAL_SECONDS", 2.0),
            timeout_seconds=optional_int("DPE_UNLIMITED_OCR_TIMEOUT_SECONDS", 180),
        )


class UnlimitedOCRAdapter:
    supports_multi_stage = False
    max_image_side = 8192

    def __init__(self, config: UnlimitedOCRConfig) -> None:
        self._config = config
        self._access_token: str | None = None

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
        layouts = parse_unlimited_ocr_layouts(parse_result)
        return OCRPageResult(
            layouts=layouts,
            source="unlimited-ocr",
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


def parse_unlimited_ocr_layouts(parse_result: dict[str, Any]) -> list[Layout]:
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
                    ref=legacy_ref_for_kind(kind, layout_type or "unlimited-ocr"),
                    det=det,
                    text=text,
                    kind=kind,
                    type=layout_type,
                    polygon=_parse_polygon(item.get("polygon")),
                    html=html,
                    source="unlimited-ocr",
                    raw=item if isinstance(item, dict) else None,
                )
            )
    return layouts


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
