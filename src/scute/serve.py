"""``scute-serve``: ``vllm serve`` with the patches on, set up from the adapters.

Usage::

    scute-serve MODEL --enable-lora --lora-modules a=/path/a b=/path/b [vllm serve args]

The adapters named in ``--lora-modules`` must be compiled (``scute-compile``).  The
engine settings are read from them, passed to vLLM's processes in the environment,
and the plugin of this package patches each process.  Every other argument goes to
``vllm serve`` unchanged.
"""

from __future__ import annotations

import json
import os
import sys

from .engine import ENV_ENABLE, ENV_OFT_BLOCK, ENV_SHARED, engine_settings


def _lora_paths(argv: list[str]) -> list[str]:
    paths, on = [], False
    for a in argv:
        if a.startswith("--lora-modules"):
            on = True
            if "=" in a and a.startswith("--lora-modules="):
                a = a.split("=", 1)[1]
            else:
                continue
        elif a.startswith("-"):
            on = False
            continue
        if on:
            if a.lstrip().startswith("{"):
                paths.append(json.loads(a)["path"])
            elif "=" in a:
                paths.append(a.split("=", 1)[1])
    return paths


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    s = engine_settings(_lora_paths(argv))
    env = dict(os.environ)
    env[ENV_ENABLE] = "1"
    env[ENV_SHARED] = s["shared_factors"] or ""
    env[ENV_OFT_BLOCK] = str(s["oft_block"] or "")
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    print(f"scute-serve: shared factors {s['shared_factors']}, OFT block {s['oft_block']}",
          file=sys.stderr)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *argv]
    os.execve(sys.executable, cmd, env)


if __name__ == "__main__":
    main()
