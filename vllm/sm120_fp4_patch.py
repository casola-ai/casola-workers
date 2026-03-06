"""
Monkey-patch: SM120 (RTX 5090) -> SM100 for vLLM's FP4 capability detection.

vLLM's FP4 check only recognizes is_device_capability_family(100)
(data-center Blackwell B100/B200). RTX 5090 reports SM120 (RTX Blackwell /
GB202) which has identical FP4 tensor-core capabilities but is not recognized,
causing a ~30-50% throughput regression via Marlin software fallback.

Installed as a .pth import in site-packages so it runs before any user code.
The actual patch is deferred until torch.cuda is first imported.

Upstream: https://github.com/vllm-project/vllm/issues/30135
Tracking: https://github.com/casola-ai/casola/issues/700
"""

import builtins
import sys

_real_import = builtins.__import__


def _apply_patch():
    cuda = sys.modules.get("torch.cuda")
    if cuda is None or getattr(cuda, "_sm120_patched", False):
        return
    orig = cuda.get_device_capability

    def get_device_capability(device=None):
        major, minor = orig(device)
        if major == 12:  # SM120 -> SM100
            return (10, minor)
        return (major, minor)

    cuda.get_device_capability = get_device_capability
    cuda._sm120_patched = True


def _import_hook(name, *args, **kwargs):
    result = _real_import(name, *args, **kwargs)
    if "torch.cuda" in sys.modules:
        builtins.__import__ = _real_import
        _apply_patch()
    return result


builtins.__import__ = _import_hook
