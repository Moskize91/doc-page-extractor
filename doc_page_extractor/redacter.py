from typing import Any, Generator, Iterable, cast

from PIL import Image, ImageDraw


def redact(
    image: Image.Image,
    fill_color: tuple[int, int, int],
    rectangles: Iterable[tuple[int, int, int, int]],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in rectangles:
        draw.rectangle((x1, y1, x2, y2), fill=fill_color)
    return image


class _AveragingColor:
    def __init__(self) -> None:
        self._r: float = 0.0
        self._g: float = 0.0
        self._b: float = 0.0
        self._a: float = 0.0
        self._count: int = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def average(self) -> tuple[float, float, float, float]:
        if self._count == 0:
            return 1.0, 1.0, 1.0, 1.0
        return (
            self._r / self._count,
            self._g / self._count,
            self._b / self._count,
            self._a / self._count,
        )

    def add_color(self, r: float, g: float, b: float, a: float) -> None:
        self._r += r
        self._g += g
        self._b += b
        self._a += a
        self._count += 1


def background_color(image: Image.Image) -> tuple[int, int, int]:
    """将像素颜色按灰度排序，取中位颜色。此颜色与纸张的颜色相同，可做背景色"""
    pixels_count = image.width * image.height
    if pixels_count == 0:
        return 255, 255, 255

    bucket: list[_AveragingColor | None] = [None] * 256
    for r, g, b, a in _iter_pixels(image):
        gray = round(255 * _gray(r, g, b, a))
        colors = bucket[gray]
        if colors is None:
            colors = _AveragingColor()
            bucket[gray] = colors
        colors.add_color(r, g, b, a)

    offset: int = 0
    found_colors: _AveragingColor | None = None

    for colors in bucket:
        if not colors:
            continue
        offset += colors.count
        if offset > pixels_count // 2:
            found_colors = colors
            break

    assert found_colors is not None
    r, g, b, a = found_colors.average

    # 背景色为白色
    r = r * a + 1.0 * (1.0 - a)
    g = g * a + 1.0 * (1.0 - a)
    b = b * a + 1.0 * (1.0 - a)

    return round(r * 255), round(g * 255), round(b * 255)


def _gray(r: float, g: float, b: float, a: float) -> float:
    # ITU-R BT.601 https://en.wikipedia.org/wiki/Rec._601
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray * a


def _iter_pixels(
    image: Image.Image,
) -> Generator[tuple[float, float, float, float], None, None]:
    for pixel in cast(Any, image.getdata()):
        pixel_len = len(cast(tuple, pixel)) if isinstance(pixel, tuple) else 1
        if pixel_len == 4:
            # RGBA 格式
            r, g, b, a = cast(tuple[int, int, int, int], pixel)
        elif pixel_len == 3:
            # RGB 格式
            r, g, b = cast(tuple[int, int, int], pixel)
            a = 255
        elif pixel_len == 2:
            # LA 格式 (灰度 + alpha)
            l, a = cast(tuple[int, int], pixel)
            r = g = b = l
        else:
            # L 格式 (灰度)
            r = g = b = cast(int, pixel)
            a = 255
        yield (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
