# pylint: disable=undefined-all-variable

__version__ = "1.2.0"

_LAZY_EXPORTS = {
    "AbortError": ("extraction_context", "AbortError"),
    "DeepSeekOCR2VendorAdapter": ("adapters", "DeepSeekOCR2VendorAdapter"),
    "DeepSeekOCR2VendorConfig": ("adapters", "DeepSeekOCR2VendorConfig"),
    "DeepSeekBackend": ("types", "DeepSeekBackend"),
    "DeepSeekOCRSize": ("types", "DeepSeekOCRSize"),
    "DeepSeekOCRVendorAdapter": ("adapters", "DeepSeekOCRVendorAdapter"),
    "DeepSeekOCRVendorConfig": ("adapters", "DeepSeekOCRVendorConfig"),
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
    "UnlimitedModelOCRAdapter": ("adapters", "UnlimitedModelOCRAdapter"),
    "UnlimitedOCRVendorAdapter": ("adapters", "UnlimitedOCRVendorAdapter"),
    "UnlimitedOCRVendorConfig": ("adapters", "UnlimitedOCRVendorConfig"),
    "create_deepseek_ocr_page_extractor": ("extractor", "create_deepseek_ocr_page_extractor"),
    "create_page_extractor_with_adapter": ("extractor", "create_page_extractor_with_adapter"),
    "create_deepseek_ocr2_vendor_page_extractor": ("extractor", "create_deepseek_ocr2_vendor_page_extractor"),
    "create_deepseek_ocr_vendor_page_extractor": ("extractor", "create_deepseek_ocr_vendor_page_extractor"),
    "create_unlimited_ocr_page_extractor": ("extractor", "create_unlimited_ocr_page_extractor"),
    "create_unlimited_ocr_vendor_page_extractor": ("extractor", "create_unlimited_ocr_vendor_page_extractor"),
    "plot": ("plot", "plot"),
}

__all__ = [
    "plot",
    "create_deepseek_ocr_page_extractor",
    "create_page_extractor_with_adapter",
    "create_deepseek_ocr_vendor_page_extractor",
    "create_deepseek_ocr2_vendor_page_extractor",
    "create_unlimited_ocr_page_extractor",
    "create_unlimited_ocr_vendor_page_extractor",
    "PageExtractor",
    "OCRAdapter",
    "OCRPageResult",
    "DeepSeekOCRSize",
    "DeepSeekBackend",
    "DeepSeekOCRVendorConfig",
    "DeepSeekOCRVendorAdapter",
    "DeepSeekOCR2VendorConfig",
    "DeepSeekOCR2VendorAdapter",
    "UnlimitedOCRVendorConfig",
    "UnlimitedOCRVendorAdapter",
    "UnlimitedModelOCRAdapter",
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
