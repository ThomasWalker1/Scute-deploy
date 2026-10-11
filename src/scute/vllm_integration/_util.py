"""Patch bookkeeping, so the integration can be undone and audited.

Every patch this package applies goes through :func:`patch_attr`, which records the
original.  :func:`applied_patches` lists exactly what was changed in the engine, and
:func:`revert_all` returns to stock vLLM.
"""

from __future__ import annotations

from typing import Any

#: ``(owner, name, original_value, was_present)`` in application order.
_PATCHES: list[tuple[Any, str, Any, bool]] = []


def patch_attr(owner: Any, name: str, value: Any) -> None:
    """Replace ``owner.name`` with ``value``, remembering what was there."""
    was_present = name in getattr(owner, "__dict__", {}) or hasattr(owner, name)
    original = getattr(owner, name, None)
    _PATCHES.append((owner, name, original, was_present))
    setattr(owner, name, value)


def applied_patches() -> list[str]:
    """Human-readable list of every patched symbol."""
    return [f"{getattr(o, '__name__', o)}.{n}" for o, n, _, _ in _PATCHES]


def revert_all() -> None:
    """Undo every patch, most recent first."""
    while _PATCHES:
        owner, name, original, was_present = _PATCHES.pop()
        if was_present:
            setattr(owner, name, original)
        else:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
