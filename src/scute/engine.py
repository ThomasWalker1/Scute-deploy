"""Turn the engine on: read the compiled adapters, set the engine settings, patch vLLM.

Two settings hold for a whole engine, and both come from the compiled adapters:

* the VeRA shared factors, which every VeRA adapter in the engine must share;
* the OFT block size, which every OFT adapter in the engine must use.

They are passed to every vLLM process through environment variables, and the vLLM
plugin of this package (:mod:`scute.plugin`) applies the patches in each process.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

from .compile import SHARED_FILE

ENV_ENABLE = "SCUTE_ENABLE"
ENV_SHARED = "SCUTE_SHARED_FACTORS"
ENV_OFT_BLOCK = "SCUTE_OFT_BLOCK"


def engine_settings(adapter_dirs: Iterable[str]) -> dict:
    """The engine settings that a set of compiled adapters needs.

    Returns ``{"shared_factors": path or None, "oft_block": int or None}``.

    Raises:
        ValueError: if a directory is not a compiled adapter, if VeRA adapters use
            different shared factors, or if OFT adapters use different block sizes.
    """
    shared: dict[str, str] = {}
    blocks: set[int] = set()
    for d in adapter_dirs:
        cfg = json.load(open(os.path.join(d, "adapter_config.json")))
        info = cfg.get("scute")
        if info is None:
            raise ValueError(f"{d} is not a compiled adapter: run scute.compile_adapter first")
        if info["method"] == "vera":
            shared.setdefault(info["shared_key"], os.path.join(os.path.abspath(d), SHARED_FILE))
        elif info["method"] == "oft":
            blocks.add(int(info["block"]))
    if len(shared) > 1:
        raise ValueError(f"VeRA adapters with different shared factors: {sorted(shared)}. "
                         "One engine serves one pair of shared factors.")
    if len(blocks) > 1:
        raise ValueError(f"OFT adapters with different block sizes: {sorted(blocks)}. "
                         "One engine serves one block size.")
    return dict(shared_factors=next(iter(shared.values()), None),
                oft_block=next(iter(blocks), None))


def enable(adapters: Iterable[str] | None = None, *, shared_factors: str | None = None,
           oft_block: int | None = None, allow_version_mismatch: bool = False) -> dict:
    """Patch vLLM so that it serves LoRA, DoRA, (IA)^3, VeRA and OFT in one batch.

    Call this before you create ``vllm.LLM``.

    Args:
        adapters: the compiled adapter directories that the engine will serve.  The
            engine settings are read from them.
        shared_factors: the VeRA shared factors, ``scute_shared.safetensors``.  This
            overrides the value read from ``adapters``.
        oft_block: the OFT block size.  This overrides the value read from ``adapters``.
        allow_version_mismatch: proceed on a vLLM release that was not checked.

    Returns the engine settings in use.
    """
    settings = engine_settings(adapters or [])
    if shared_factors is not None:
        settings["shared_factors"] = os.path.abspath(shared_factors)
    if oft_block is not None:
        settings["oft_block"] = int(oft_block)

    from . import vllm_integration

    wanted = {ENV_SHARED: settings["shared_factors"] or "",
              ENV_OFT_BLOCK: str(settings["oft_block"] or "")}
    if vllm_integration.is_enabled():
        current = {k: os.environ.get(k, "") for k in wanted}
        if current != wanted:
            raise RuntimeError(
                f"vLLM is already patched with {current}; these adapters need {wanted}. "
                "Start a new process for a different set of engine settings.")
        return settings

    os.environ.update(wanted)
    os.environ[ENV_ENABLE] = "1"
    # A forked engine process can inherit a lock held by a torch thread of this process
    # (for example after compile_adapter) and wait on it forever.  A spawned process
    # starts clean, and the plugin patches it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    vllm_integration.apply_patches(allow_version_mismatch=allow_version_mismatch)
    return settings
