"""Compatibility wrapper for older imports.

ComfyUI loads this custom node through __init__.py, which imports tjs_nodes.
This file re-exports the same mappings in case a local workflow imports
ComfyUI_TJS.nodes directly.
"""

try:
    from .tjs_nodes import (
        NODE_CLASS_MAPPINGS as _SAMPLER_CLASS,
        NODE_DISPLAY_NAME_MAPPINGS as _SAMPLER_DISPLAY,
        TJSAdvancedSampler,
        TJSCustom,
        TJSCustomAdvanced,
        TJSSampler,
    )
except ImportError:
    from tjs_nodes import (  # type: ignore
        NODE_CLASS_MAPPINGS as _SAMPLER_CLASS,
        NODE_DISPLAY_NAME_MAPPINGS as _SAMPLER_DISPLAY,
        TJSAdvancedSampler,
        TJSCustom,
        TJSCustomAdvanced,
        TJSSampler,
    )

NODE_CLASS_MAPPINGS = {**_SAMPLER_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**_SAMPLER_DISPLAY}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TJSAdvancedSampler",
    "TJSCustom",
    "TJSCustomAdvanced",
    "TJSSampler",
]
