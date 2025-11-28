from typing import Iterable, cast

from PIL import ImageDraw
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont, load_default

from .types import Layout

_FRAGMENT_COLOR = (0x49, 0xCF, 0xCB)  # Light Green
_Color = tuple[int, int, int]


def plot(image: Image, layouts: Iterable[Layout]) -> Image:
    layout_font = cast(FreeTypeFont, load_default(size=35))
    draw = ImageDraw.Draw(image, mode="RGBA")

    def _draw_text(
        position: tuple[int, int],
        text: str,
        font: FreeTypeFont,
        bold: bool,
        color: _Color,
    ) -> None:
        nonlocal draw
        x, y = position
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        offset = round(font.size * 0.15)

        for dx, dy in _generate_delta(bold):
            draw.text(
                xy=(x + dx - text_width - offset, y + dy),
                text=text,
                font=font,
                fill=color,
            )

    for layout in layouts:
        x1, y1, x2, y2 = layout.det
        draw.polygon(
            xy=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            outline=_FRAGMENT_COLOR,
            width=5,
        )

    for i, layout in enumerate(layouts):
        x1, y1, _, _ = layout.det
        _draw_text(
            position=(x1, y1),
            text=f"{i + 1}. {layout.ref.strip()}",
            font=layout_font,
            bold=True,
            color=_FRAGMENT_COLOR,
        )
    return image


def _generate_delta(bold: bool):
    if bold:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                yield dx, dy
    else:
        yield 0, 0
