# pylint: disable=undefined-all-variable

__version__ = "1.1.2"

_LAZY_EXPORTS = {
    "AbortError": ("extraction_context", "AbortError"),
    "BaiduCloudOCRAdapter": ("adapters", "BaiduCloudOCRAdapter"),
    "BaiduCloudOCRConfig": ("adapters", "BaiduCloudOCRConfig"),
    "DeepSeekLocalOCRAdapter": ("adapters", "DeepSeekLocalOCRAdapter"),
    "DeepSeekModelOCRAdapter": ("adapters", "DeepSeekModelOCRAdapter"),
    "DeepSeekOCRModel": ("types", "DeepSeekOCRModel"),
    "DeepSeekOCRSize": ("types", "DeepSeekOCRSize"),
    "DeepSeekVendorOCRAdapter": ("adapters", "DeepSeekVendorOCRAdapter"),
    "DeepSeekVendorOCRConfig": ("adapters", "DeepSeekVendorOCRConfig"),
    "ExtractionAbortedError": ("extraction_context", "ExtractionAbortedError"),
    "ExtractionContext": ("types", "ExtractionContext"),
    "Layout": ("types", "Layout"),
    "LayoutKind": ("types", "LayoutKind"),
    "OCRAdapter": ("types", "OCRAdapter"),
    "OCRPageResult": ("types", "OCRPageResult"),
    "PageBlock": ("types", "PageBlock"),
    "PageExtractor": ("types", "PageExtractor"),
    "StructuredPage": ("types", "StructuredPage"),
    "TokenLimitError": ("extraction_context", "TokenLimitError"),
    "create_baidu_page_extractor": ("extractor", "create_baidu_page_extractor"),
    "create_page_extractor": ("extractor", "create_page_extractor"),
    "create_page_extractor_with_adapter": ("extractor", "create_page_extractor_with_adapter"),
    "create_page_extractor_with_model": ("extractor", "create_page_extractor_with_model"),
    "create_deepseek_vendor_page_extractor": ("extractor", "create_deepseek_vendor_page_extractor"),
    "plot": ("plot", "plot"),
}

__all__ = [
    "plot",
    "create_baidu_page_extractor",
    "create_page_extractor",
    "create_page_extractor_with_adapter",
    "create_page_extractor_with_model",
    "create_deepseek_vendor_page_extractor",
    "PageExtractor",
    "OCRAdapter",
    "OCRPageResult",
    "DeepSeekOCRSize",
    "DeepSeekOCRModel",
    "DeepSeekVendorOCRConfig",
    "DeepSeekVendorOCRAdapter",
    "DeepSeekModelOCRAdapter",
    "DeepSeekLocalOCRAdapter",
    "BaiduCloudOCRConfig",
    "BaiduCloudOCRAdapter",
    "ExtractionContext",
    "AbortError",
    "ExtractionAbortedError",
    "TokenLimitError",
    "Layout",
    "LayoutKind",
    "PageBlock",
    "StructuredPage",
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
