import re
from enum import Enum, auto
from typing import Generator

_TAG_PATTERN = re.compile(r"<\|(det|ref)\|>(.+?)<\|/\1\|>")
_DET_COORDS_PATTERN = re.compile(r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]")


class ParsedItemKind(Enum):
    DET = auto()
    REF = auto()
    TEXT = auto()


ParsedItem = (
    tuple[ParsedItemKind.DET, tuple[int, int, int, int]]
    | tuple[ParsedItemKind.REF, str]
    | tuple[ParsedItemKind.TEXT, str]
)


def parse_ocr_response(
    response: str, width: int, height: int
) -> Generator[ParsedItem, None, None]:
    last_end: int = 0
    for matched in _TAG_PATTERN.finditer(response):
        if matched.start() > last_end:
            plain_text = response[last_end : matched.start()]
            if plain_text:
                yield ParsedItemKind.TEXT, plain_text
        tag_type = matched.group(1)
        content = matched.group(2)
        if tag_type == "det":
            coords_match = _DET_COORDS_PATTERN.search(content)
            if coords_match:
                x1_norm, y1_norm, x2_norm, y2_norm = [
                    int(c) for c in coords_match.groups()
                ]
                x1 = round(x1_norm / 1000 * width)
                y1 = round(y1_norm / 1000 * height)
                x2 = round(x2_norm / 1000 * width)
                y2 = round(y2_norm / 1000 * height)
                yield ParsedItemKind.DET, (x1, y1, x2, y2)
        elif tag_type == "ref":
            yield ParsedItemKind.REF, content
        last_end = matched.end()

    if last_end < len(response):
        plain_text = response[last_end:]
        if plain_text:
            yield ParsedItemKind.TEXT, plain_text
