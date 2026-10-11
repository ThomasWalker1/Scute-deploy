"""Triton kernels for the routes inside vLLM's LoRA path.

* :mod:`.vllm_expand_s`: vLLM's ``lora_expand`` with the output scale ``s`` applied as
  each tile finishes.
* :mod:`.vllm_shrink_m`: vLLM's ``lora_shrink`` that skips adapters without a term.
* :mod:`.vllm_shared`: the shared-factor term of VeRA, as GEMMs over row tiles.
* :mod:`.vllm_pretransform`: OFT's block-diagonal input transform before the base GEMM.
* :mod:`.pretransform`: OFT's load-time fold (the Cayley map) and a float64 reference.
"""
