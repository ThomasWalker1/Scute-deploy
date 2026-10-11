"""Let vLLM's PEFT config validator accept ``use_dora``.

vLLM refuses a DoRA checkpoint in ``PEFTHelper._validate_features``, which appends
``"vLLM does not yet support DoRA."`` to an error list.  That refusal is the visible
symptom; the loader behind it also cannot carry the magnitude tensor, which
``_loading`` handles.

Only the DoRA clause is lifted.  Every other unsupported-feature check --
``modules_to_save`` in particular -- keeps refusing, because nothing here makes those
work and a checkpoint that silently loads wrong is worse than one that is refused.
"""

from __future__ import annotations

from ._util import patch_attr


def apply() -> None:
    from vllm.lora.peft_helper import PEFTHelper

    original = PEFTHelper._validate_features

    def _validate_features(self) -> list[str]:
        errors = original(self)
        return [e for e in errors if "DoRA" not in e]

    patch_attr(PEFTHelper, "_validate_features", _validate_features)
