"""Engine settings read from compiled adapters, and the server's argument parsing."""

from __future__ import annotations

import json
import os

import pytest

from scute.engine import engine_settings
from scute.serve import _lora_paths


def _compiled(tmp, name, method, **extra):
    d = tmp / name
    d.mkdir()
    json.dump(dict(peft_type="LORA", scute=dict(method=method, **extra)),
              open(d / "adapter_config.json", "w"))
    return str(d)


def test_settings_from_adapters(tmp_path):
    dirs = [_compiled(tmp_path, "l", "lora"), _compiled(tmp_path, "v", "vera", shared_key="k1"),
            _compiled(tmp_path, "w", "vera", shared_key="k1"),
            _compiled(tmp_path, "o", "oft", block=32)]
    s = engine_settings(dirs)
    assert s["oft_block"] == 32
    assert s["shared_factors"] == os.path.join(os.path.abspath(dirs[1]), "scute_shared.safetensors")


def test_lora_only_needs_no_settings(tmp_path):
    assert engine_settings([_compiled(tmp_path, "l", "lora")]) == dict(
        shared_factors=None, oft_block=None)


def test_two_shared_pairs_are_refused(tmp_path):
    dirs = [_compiled(tmp_path, "v", "vera", shared_key="k1"),
            _compiled(tmp_path, "w", "vera", shared_key="k2")]
    with pytest.raises(ValueError, match="shared factors"):
        engine_settings(dirs)


def test_two_block_sizes_are_refused(tmp_path):
    dirs = [_compiled(tmp_path, "o", "oft", block=32), _compiled(tmp_path, "p", "oft", block=16)]
    with pytest.raises(ValueError, match="block sizes"):
        engine_settings(dirs)


def test_uncompiled_adapter_is_refused(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    json.dump(dict(peft_type="IA3"), open(d / "adapter_config.json", "w"))
    with pytest.raises(ValueError, match="not a compiled adapter"):
        engine_settings([str(d)])


def test_server_reads_lora_modules():
    argv = ["m", "--port", "8000", "--lora-modules", "a=/x/a", "b=/x/b", "--dtype", "float16",
            "--lora-modules={\"name\": \"c\", \"path\": \"/x/c\"}"]
    assert _lora_paths(argv) == ["/x/a", "/x/b", "/x/c"]
