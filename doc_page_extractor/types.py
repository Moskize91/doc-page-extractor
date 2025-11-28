from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from typing import Generator, Literal, Callable

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
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0

@runtime_checkable
class PageExtractor(Protocol):
    def download_models(self) -> None:
        ...

    def load_models(self) -> None:
        ...

    def extract(
        self,
        image: Image.Image,
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
    ) -> Generator[tuple[Image.Image, list[Layout]], None, None]:
        ...