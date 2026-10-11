"""Scute: serve LoRA, DoRA, (IA)^3, VeRA and OFT adapters exactly, in one vLLM batch.

Two steps::

    import scute
    scute.compile_adapter("peft/my-ia3", "compiled/my-ia3", "meta-llama/Llama-3.1-8B")
    scute.enable(["compiled/my-ia3", ...])     # before vllm.LLM(...)

Then serve the compiled directories with ``vllm.lora.request.LoRARequest`` as usual.
"""

from .compile import Compiled, compile_adapter
from .engine import enable, engine_settings

__version__ = "0.1.0"

__all__ = ["compile_adapter", "Compiled", "enable", "engine_settings", "__version__"]
