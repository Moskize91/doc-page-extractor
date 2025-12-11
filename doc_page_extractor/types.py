from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import runtime_checkable, Protocol, Generator, Literal, Callable

from PIL import Image


DeepSeekOCRSize = Literal["tiny", "small", "base", "large", "gundam"]


@dataclass
class Layout:
    ref: str
    det: tuple[int, int, int, int]
    text: str | None


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
        image: Image.Image,
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple[Image.Image, list[Layout]], None, None]:
        ...


@runtime_checkable
class DeepSeekOCRModel(Protocol):
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
