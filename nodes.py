"""Compatibility wrapper for older imports.

ComfyUI loads this custom node through __init__.py, which imports tjs_nodes.
This file re-exports the same mappings in case a local workflow imports
ComfyUI_TJS.nodes directly.
"""

try:
    from .tjs_nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        TJSDecode,
        TJSDecodeManualSigma,
        TJSSampler,
    )
except ImportError:
    from tjs_nodes import (  # type: ignore
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        TJSDecode,
        TJSDecodeManualSigma,
        TJSSampler,
    )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TJSDecode",
    "TJSDecodeManualSigma",
    "TJSSampler",
]
