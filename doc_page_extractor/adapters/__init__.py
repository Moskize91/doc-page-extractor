# pylint: disable=undefined-all-variable

_LAZY_EXPORTS = {
    "GLMOCRServiceAdapter": ("glmocr", "GLMOCRServiceAdapter"),
    "GLMOCRServiceConfig": ("glmocr", "GLMOCRServiceConfig"),
    "parse_glm_ocr_layouts": ("glmocr", "parse_glm_ocr_layouts"),
    "DeepSeekOCR2VendorAdapter": ("deepseek", "DeepSeekOCR2VendorAdapter"),
    "DeepSeekOCR2VendorConfig": ("deepseek", "DeepSeekOCR2VendorConfig"),
    "DeepSeekOCRVendorAdapter": ("deepseek", "DeepSeekOCRVendorAdapter"),
    "DeepSeekOCRVendorConfig": ("deepseek", "DeepSeekOCRVendorConfig"),
    "UnlimitedModelOCRAdapter": ("unlimited", "UnlimitedModelOCRAdapter"),
    "UnlimitedOCRVendorAdapter": ("unlimited", "UnlimitedOCRVendorAdapter"),
    "UnlimitedOCRVendorConfig": ("unlimited", "UnlimitedOCRVendorConfig"),
    "parse_unlimited_ocr_layouts": ("unlimited", "parse_unlimited_ocr_layouts"),
    "parse_unlimited_ocr_local_layouts": ("unlimited", "parse_unlimited_ocr_local_layouts"),
    "parse_deepseek_ocr2_layouts": ("deepseek", "parse_deepseek_ocr2_layouts"),
    "parse_deepseek_ocr_layouts": ("deepseek", "parse_deepseek_ocr_layouts"),
}

__all__ = [
    "GLMOCRServiceAdapter",
    "GLMOCRServiceConfig",
    "parse_glm_ocr_layouts",
    "DeepSeekOCR2VendorAdapter",
    "DeepSeekOCR2VendorConfig",
    "DeepSeekOCRVendorAdapter",
    "DeepSeekOCRVendorConfig",
    "UnlimitedModelOCRAdapter",
    "UnlimitedOCRVendorAdapter",
    "UnlimitedOCRVendorConfig",
    "parse_unlimited_ocr_layouts",
    "parse_unlimited_ocr_local_layouts",
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
