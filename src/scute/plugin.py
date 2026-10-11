"""vLLM general plugin: apply the patches in every vLLM process.

vLLM loads the plugins of the ``vllm.general_plugins`` entry-point group in each of
its processes: the API server, the engine core and every worker.  This plugin patches
the process when ``SCUTE_ENABLE=1``, and does nothing otherwise, so an installed
package leaves stock vLLM unchanged.
"""

from __future__ import annotations

import os


def register() -> None:
    if os.environ.get("SCUTE_ENABLE") != "1":
        return
    from .vllm_integration import apply_patches

    apply_patches(allow_version_mismatch=os.environ.get("SCUTE_ALLOW_VERSION_MISMATCH") == "1")
