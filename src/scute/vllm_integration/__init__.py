"""Patch an unmodified vLLM at run time so that it serves every compiled method.

vLLM is installed from a wheel and never edited.  :func:`apply_patches` replaces a
small set of private vLLM symbols, and :func:`applied_patches` lists them.  The
scheduler, the paging path, attention and the batch composition are not changed.

The patches:

* ``_config``: vLLM's PEFT validator accepts ``use_dora``.
* ``_loading``: the adapter loader carries DoRA's magnitude and the normal-form
  tensors (``scute_s``, ``scute_d``, ``scute_e``, ``scute_R``).
* ``_fold``: at adapter activation, DoRA's scale is folded and every slot is written
  into per-layer pools, allocated with the LoRA weights so that CUDA graphs and
  ``torch.compile`` see them from the first trace.
* ``_expand``: each LoRA layer runs the routes: the OFT input transform before the
  base GEMM, vLLM's expand with the scale ``s`` in its epilogue, and the VeRA
  shared-factor term.

The patched symbols are private and move between vLLM releases, so the version is
checked, and an unchecked release fails at setup.  Use :func:`scute.enable` rather
than this module directly.
"""

from __future__ import annotations

import os

from ._guards import UnsupportedDoRAPath, check_config, check_module
from ._util import applied_patches, revert_all

__all__ = [
    "apply_patches",
    "applied_patches",
    "revert",
    "is_enabled",
    "UnsupportedDoRAPath",
    "check_config",
    "check_module",
    "SUPPORTED_VLLM",
]

#: Releases this integration has been checked against.
SUPPORTED_VLLM = ("0.19.1",)

_ENABLED = False


def apply_patches(*, allow_version_mismatch: bool = False) -> None:
    """Patch vLLM in this process.  Calling it again does nothing.

    The engine settings are read from the environment when the patches are applied:
    ``SCUTE_SHARED_FACTORS`` (the VeRA shared factors) and ``SCUTE_OFT_BLOCK`` (the OFT
    block size).  :func:`scute.enable` sets both from the compiled adapters.

    Raises:
        ValueError: on a vLLM release that is not in :data:`SUPPORTED_VLLM`.
    """
    global _ENABLED
    if _ENABLED:
        return

    import vllm

    if vllm.__version__ not in SUPPORTED_VLLM and not allow_version_mismatch:
        raise ValueError(
            f"vllm {vllm.__version__} is not in {SUPPORTED_VLLM}. This integration "
            "patches private symbols; check them against the new release first."
        )

    # vLLM caches compiled graphs on disk under a key built from the engine config.
    # The key does not see these patches, so a graph compiled by an unpatched engine
    # could be loaded and would compute the wrong function.
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    from . import _config, _expand, _fold, _loading

    _config.apply()
    _loading.apply()
    _fold.apply()
    _expand.apply()
    _ENABLED = True


def revert() -> None:
    """Undo every patch and return to stock vLLM."""
    global _ENABLED
    revert_all()
    _ENABLED = False


def is_enabled() -> bool:
    """True if vLLM is patched in this process."""
    return _ENABLED
