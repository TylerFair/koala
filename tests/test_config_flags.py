import ast
import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
import yaml

import fit_jwst
from tools.notebook_demo_runtime import apply_bounded_demo_runtime


ROOT = Path(__file__).resolve().parents[1]


def _literal_flag_keys():
    """Return literal keys used through the local ``flags`` mapping."""
    keys = set()
    paths = [ROOT / "fit_jwst.py", *(ROOT / "koala").rglob("*.py")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "flags"
                and node.func.attr in {"get", "setdefault", "pop"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "flags"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
    return keys


def _flag_mappings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "flags" and isinstance(child, dict):
                yield child
            yield from _flag_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flag_mappings(child)


def _markdown_yaml_documents():
    for path in sorted((ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"```yaml\s*\n(.*?)```", text, re.DOTALL):
            payload = yaml.safe_load(match.group(1))
            if payload is not None:
                yield path, payload


def _printed_literal_text():
    paths = [
        ROOT / "fit_jwst.py",
        *(ROOT / "koala").rglob("*.py"),
        *(ROOT / "models").rglob("*.py"),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                continue
            yield " ".join(
                child.value
                for argument in node.args
                for child in ast.walk(argument)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            )


def test_flag_tiers_are_disjoint_and_cover_every_literal_lookup():
    assert set(fit_jwst.FLAG_TIERS) == {"public", "advanced", "internal"}
    tier_total = sum(len(keys) for keys in fit_jwst.FLAG_TIERS.values())
    assert tier_total == len(fit_jwst.KNOWN_FLAGS)
    assert _literal_flag_keys() <= fit_jwst.KNOWN_FLAGS


def test_unknown_flag_warns_with_closest_match_and_does_not_fail():
    with pytest.warns(
        fit_jwst.UnknownFlagWarning,
        match=r"detrendng_type.*detrending_type",
    ):
        fit_jwst._validate_flag_keys({"detrendng_type": "linear"})


def test_all_known_keys_are_silent_including_legacy_stage_overrides():
    configured = {key: None for key in fit_jwst.KNOWN_FLAGS}
    with warnings.catch_warnings():
        warnings.simplefilter("error", fit_jwst.UnknownFlagWarning)
        fit_jwst._validate_flag_keys(configured)


def test_configuration_guide_tracks_the_public_surface_only():
    guide = (ROOT / "docs/guides/configuration.md").read_text(encoding="utf-8")
    for key in fit_jwst.PUBLIC_FLAGS | fit_jwst.ADVANCED_FLAGS:
        assert re.search(rf"\b{re.escape(key)}\b", guide), key
    for key in fit_jwst.INTERNAL_FLAGS:
        assert f"`{key}`" not in guide, key


def test_example_flags_are_public_or_advanced():
    visible = fit_jwst.PUBLIC_FLAGS | fit_jwst.ADVANCED_FLAGS
    for path in sorted((ROOT / "examples").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for flags in _flag_mappings(payload):
            assert set(flags) <= visible, path


def test_notebook_source_configs_do_not_enumerate_internal_flags():
    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            line
            for cell in notebook.get("cells", [])
            for line in cell.get("source", [])
        )
        for key in fit_jwst.INTERNAL_FLAGS:
            assert not re.search(
                rf"[\"']{re.escape(key)}[\"']\s*:", source
            ), (path, key)


def test_notebook_smoke_profile_is_internal_only():
    config = {"flags": {"detrending_type": "linear"}}
    original = set(config["flags"])
    assert apply_bounded_demo_runtime(config) is config
    added = set(config["flags"]) - original
    assert added
    assert {"whitelight_num_warmup", "whitelight_num_samples",
            "highres_num_warmup", "highres_num_samples"} <= added


def test_documented_yaml_flags_are_public_or_advanced():
    visible = fit_jwst.PUBLIC_FLAGS | fit_jwst.ADVANCED_FLAGS
    for path, payload in _markdown_yaml_documents():
        for flags in _flag_mappings(payload):
            assert set(flags) <= visible, path


def test_user_docs_do_not_enumerate_internal_flag_names():
    for path in sorted((ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for key in fit_jwst.INTERNAL_FLAGS:
            assert f"`{key}`" not in text, (path, key)
            assert f"`flags.{key}`" not in text, (path, key)


def test_normal_prints_do_not_name_internal_flags():
    printed_text = "\n".join(_printed_literal_text())
    for key in fit_jwst.INTERNAL_FLAGS - {"plots"}:
        assert not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])",
            printed_text,
        ), key
    assert "flags.plots" not in printed_text
    assert "plots=" not in printed_text
