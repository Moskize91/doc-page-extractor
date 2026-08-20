import re
from collections.abc import Iterable

from .types import Layout, LayoutKind, PageBlock, StructuredPage

_ASSET_KINDS = {
    LayoutKind.IMAGE,
    LayoutKind.TABLE,
    LayoutKind.EQUATION,
}

_CAPTION_TO_ASSET = {
    LayoutKind.IMAGE_CAPTION: LayoutKind.IMAGE,
    LayoutKind.TABLE_CAPTION: LayoutKind.TABLE,
    LayoutKind.EQUATION_CAPTION: LayoutKind.EQUATION,
}

_IGNORED_KINDS = {
    LayoutKind.HEADER,
    LayoutKind.FOOTER,
    LayoutKind.PAGE_NUMBER,
    LayoutKind.ASIDE,
}

_TABLE_CAPTION_PATTERNS = (
    re.compile(r"^\s*第\s*[一二三四五六七八九十百千万零〇\d\s]+\s*表\b"),
    re.compile(r"^\s*表\s*[一二三四五六七八九十百千万零〇\d\s]+\b"),
    re.compile(r"^\s*table\s+\d+\b", re.IGNORECASE),
)

_FOOTNOTE_MARK_PATTERN = re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*\S+")


def deepseek_ref_to_kind(ref: str | None, text: str | None = None) -> LayoutKind:
    normalized = (ref or "").strip()
    if normalized in {"text", "正文"}:
        return LayoutKind.TEXT
    if normalized in {"title", "sub_title"}:
        return LayoutKind.TITLE
    if normalized in {"image", "figure"}:
        return LayoutKind.IMAGE
    if normalized in {"image_caption", "figure_caption"}:
        return LayoutKind.IMAGE_CAPTION
    if normalized == "figure_title":
        if _looks_like_table_caption(text):
            return LayoutKind.TABLE_CAPTION
        return LayoutKind.IMAGE_CAPTION
    if normalized in {"table"}:
        return LayoutKind.TABLE
    if normalized in {"table_caption"}:
        return LayoutKind.TABLE_CAPTION
    if normalized in {"equation", "formula"}:
        return LayoutKind.EQUATION
    if normalized in {"equation_caption", "formula_caption"}:
        return LayoutKind.EQUATION_CAPTION
    if normalized in {"footnote"}:
        return LayoutKind.FOOTNOTE
    return LayoutKind.UNKNOWN


def unlimited_ocr_type_to_kind(
    layout_type: str | None, text: str | None = None
) -> LayoutKind:
    normalized = (layout_type or "").strip()
    if normalized == "text":
        if _looks_like_table_caption(text):
            return LayoutKind.TABLE_CAPTION
        return LayoutKind.TEXT
    if normalized == "paragraph_title":
        return LayoutKind.TITLE
    if normalized == "formula":
        return LayoutKind.EQUATION
    if normalized == "image":
        return LayoutKind.IMAGE
    if normalized == "figure_title":
        return LayoutKind.IMAGE_CAPTION
    if normalized == "table":
        return LayoutKind.TABLE
    if normalized == "footnote":
        return LayoutKind.FOOTNOTE
    if normalized == "header":
        return LayoutKind.HEADER
    if normalized == "footer":
        return LayoutKind.FOOTER
    if normalized == "number":
        if _FOOTNOTE_MARK_PATTERN.search(text or ""):
            return LayoutKind.FOOTNOTE
        return LayoutKind.PAGE_NUMBER
    if normalized == "aside_text":
        return LayoutKind.ASIDE
    return LayoutKind.UNKNOWN


def legacy_ref_for_kind(kind: LayoutKind, fallback: str | None = None) -> str:
    if kind == LayoutKind.TITLE:
        return "sub_title"
    if kind == LayoutKind.FOOTNOTE:
        return "text"
    if kind == LayoutKind.PAGE_NUMBER:
        return "text"
    if kind in _IGNORED_KINDS:
        return "text"
    if kind == LayoutKind.UNKNOWN:
        return fallback or "unknown"
    return kind.value


def build_structured_page(layouts: Iterable[Layout]) -> StructuredPage:
    blocks: list[PageBlock] = []
    ignored: list[Layout] = []
    pending_asset: PageBlock | None = None
    pending_captions: dict[LayoutKind, list[PageBlock]] = {}

    for layout in layouts:
        kind = layout.kind
        if kind in _IGNORED_KINDS:
            ignored.append(layout)
            continue

        if kind in _ASSET_KINDS:
            pending_asset = _block_from_layout(layout)
            attached_captions = pending_captions.pop(kind, [])
            pending_asset.children.extend(attached_captions)
            for caption in attached_captions:
                pending_asset.layouts.extend(caption.layouts)
            blocks.append(pending_asset)
            continue

        if kind in _CAPTION_TO_ASSET:
            caption_block = _block_from_layout(layout)
            asset_kind = _CAPTION_TO_ASSET[kind]
            if pending_asset and pending_asset.kind == asset_kind:
                pending_asset.children.append(caption_block)
                pending_asset.layouts.append(layout)
            else:
                pending_captions.setdefault(asset_kind, []).append(caption_block)
            continue

        pending_asset = None
        for captions in pending_captions.values():
            blocks.extend(captions)
        pending_captions = {}
        blocks.append(_block_from_layout(layout))

    for captions in pending_captions.values():
        blocks.extend(captions)

    return StructuredPage(blocks=blocks, ignored=ignored)


def _block_from_layout(layout: Layout) -> PageBlock:
    return PageBlock(
        kind=layout.kind,
        det=layout.det,
        text=layout.text,
        html=layout.html,
        layouts=[layout],
    )


def _looks_like_table_caption(text: str | None) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _TABLE_CAPTION_PATTERNS)
