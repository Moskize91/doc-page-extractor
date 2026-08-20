from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, TYPE_CHECKING, runtime_checkable, Protocol, Generator, Literal, Callable

if TYPE_CHECKING:
    from PIL import Image


DeepSeekOCRSize = Literal["tiny", "small", "base", "large", "gundam"]
OCRModelName = Literal["deepseek-ocr", "deepseek-ocr2"]


class LayoutKind(str, Enum):
    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    EQUATION = "equation"
    EQUATION_CAPTION = "equation_caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE = "aside"
    UNKNOWN = "unknown"


@dataclass
class Layout:
    ref: str
    det: tuple[int, int, int, int]
    text: str | None
    type: str | None = None
    polygon: list[tuple[int, int]] | None = None
    html: str | None = None
    source: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)
    kind: LayoutKind = LayoutKind.UNKNOWN


@dataclass
class PageBlock:
    kind: LayoutKind
    det: tuple[int, int, int, int]
    text: str | None = None
    html: str | None = None
    layouts: list[Layout] = field(default_factory=list)
    children: list["PageBlock"] = field(default_factory=list)


@dataclass
class StructuredPage:
    blocks: list[PageBlock]
    ignored: list[Layout] = field(default_factory=list)


@dataclass
class OCRPageResult:
    layouts: list[Layout]
    source: str
    raw_text: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)
    structured: StructuredPage | None = None


@dataclass
class ExtractionContext:
    check_aborted: Callable[[], bool]
    output_dir_path: PathLike | str | None = None
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class PageExtractor(Protocol):
    def download_models(self, revision: str | None = None) -> None:
        ...

    def load_models(self) -> None:
        ...

    def extract(
        self,
        image: "Image.Image",
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple["Image.Image", list[Layout]], None, None]:
        ...

    def extract_page_results(
        self,
        image: "Image.Image",
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple["Image.Image", OCRPageResult], None, None]:
        ...


@runtime_checkable
class OCRAdapter(Protocol):
    supports_multi_stage: bool

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        ...


@runtime_checkable
class OCRModel(Protocol):
    def download(self, revision: str | None) -> None:
        ...

    def load(self) -> None:
        ...

    def unload(self) -> None:
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


@runtime_checkable
class DeepSeekOCRModel(OCRModel, Protocol):
    ...
