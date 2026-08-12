# pylint: disable=undefined-all-variable

_LAZY_EXPORTS = {
    "BaiduCloudOCRAdapter": ("baidu", "BaiduCloudOCRAdapter"),
    "BaiduCloudOCRConfig": ("baidu", "BaiduCloudOCRConfig"),
    "DeepSeekLocalOCRAdapter": ("deepseek", "DeepSeekLocalOCRAdapter"),
    "DeepSeekModelOCRAdapter": ("deepseek", "DeepSeekModelOCRAdapter"),
    "DeepSeekVendorOCRAdapter": ("deepseek", "DeepSeekVendorOCRAdapter"),
    "DeepSeekVendorOCRConfig": ("deepseek", "DeepSeekVendorOCRConfig"),
    "parse_deepseek_layouts": ("deepseek", "parse_deepseek_layouts"),
}

__all__ = [
    "BaiduCloudOCRAdapter",
    "BaiduCloudOCRConfig",
    "DeepSeekLocalOCRAdapter",
    "DeepSeekModelOCRAdapter",
    "DeepSeekVendorOCRAdapter",
    "DeepSeekVendorOCRConfig",
    "parse_deepseek_layouts",
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
