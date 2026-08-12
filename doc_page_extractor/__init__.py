# pylint: disable=undefined-all-variable

__version__ = "1.0.0"

_LAZY_EXPORTS = {
    "AbortError": ("extraction_context", "AbortError"),
    "DeepSeekOCRModel": ("types", "DeepSeekOCRModel"),
    "DeepSeekOCRSize": ("types", "DeepSeekOCRSize"),
    "ExtractionAbortedError": ("extraction_context", "ExtractionAbortedError"),
    "ExtractionContext": ("types", "ExtractionContext"),
    "Layout": ("types", "Layout"),
    "PageExtractor": ("types", "PageExtractor"),
    "TokenLimitError": ("extraction_context", "TokenLimitError"),
    "create_page_extractor": ("extractor", "create_page_extractor"),
    "create_page_extractor_with_model": ("extractor", "create_page_extractor_with_model"),
    "plot": ("plot", "plot"),
}

__all__ = [
    "plot",
    "create_page_extractor",
    "create_page_extractor_with_model",
    "PageExtractor",
    "DeepSeekOCRSize",
    "DeepSeekOCRModel",
    "ExtractionContext",
    "AbortError",
    "ExtractionAbortedError",
    "TokenLimitError",
    "Layout",
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = __import__(
        f"{__name__}.{module_name}",
        fromlist=[attribute_name],
    )
    attribute = getattr(module, attribute_name)
    globals()[name] = attribute
    return attribute
