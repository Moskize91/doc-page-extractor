# pylint: disable=undefined-all-variable

_LAZY_EXPORTS = {
    "DeepSeekOCR2VendorAdapter": ("deepseek", "DeepSeekOCR2VendorAdapter"),
    "DeepSeekOCR2VendorConfig": ("deepseek", "DeepSeekOCR2VendorConfig"),
    "DeepSeekModelOCRAdapter": ("deepseek", "DeepSeekModelOCRAdapter"),
    "DeepSeekOCRVendorAdapter": ("deepseek", "DeepSeekOCRVendorAdapter"),
    "DeepSeekOCRVendorConfig": ("deepseek", "DeepSeekOCRVendorConfig"),
    "UnlimitedOCRAdapter": ("unlimited", "UnlimitedOCRAdapter"),
    "UnlimitedOCRConfig": ("unlimited", "UnlimitedOCRConfig"),
    "parse_deepseek_ocr2_layouts": ("deepseek", "parse_deepseek_ocr2_layouts"),
    "parse_deepseek_ocr_layouts": ("deepseek", "parse_deepseek_ocr_layouts"),
}

__all__ = [
    "DeepSeekOCR2VendorAdapter",
    "DeepSeekOCR2VendorConfig",
    "DeepSeekModelOCRAdapter",
    "DeepSeekOCRVendorAdapter",
    "DeepSeekOCRVendorConfig",
    "UnlimitedOCRAdapter",
    "UnlimitedOCRConfig",
    "parse_deepseek_ocr2_layouts",
    "parse_deepseek_ocr_layouts",
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
