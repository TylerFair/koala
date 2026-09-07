"""Artifact identities, manifests, and atomic writers."""

import hashlib
import json
import os
import uuid
import jax
import numpy as np
from .constants import *

def _update_checkpoint_hash(hasher, value):
    """Add a deterministic, type-aware value encoding to ``hasher``."""
    if value is None:
        hasher.update(b"none;")
        return
    if isinstance(value, (str, bytes)):
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        hasher.update(b"text:")
        hasher.update(str(len(encoded)).encode("ascii"))
        hasher.update(b":")
        hasher.update(encoded)
        return
    if isinstance(value, (bool, int, float, np.generic)):
        hasher.update(
            f"scalar:{type(value).__name__}:{value!r};".encode("utf-8")
        )
        return
    if isinstance(value, dict):
        hasher.update(b"dict{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_checkpoint_hash(hasher, str(key))
            _update_checkpoint_hash(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(f"sequence:{type(value).__name__}[".encode("ascii"))
        for item in value:
            _update_checkpoint_hash(hasher, item)
        hasher.update(b"]")
        return
    if callable(value):
        identity = (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )
        hasher.update(f"callable:{identity};".encode("utf-8"))
        return

    try:
        array = np.asarray(jax.device_get(value))
    except Exception:
        identity = f"{type(value).__module__}.{type(value).__qualname__}"
        hasher.update(f"object:{identity};".encode("utf-8"))
        return
    contiguous = np.ascontiguousarray(array)
    hasher.update(
        (
            f"array:{contiguous.dtype.str}:"
            f"{','.join(str(size) for size in contiguous.shape)}:"
        ).encode("ascii")
    )
    hasher.update(contiguous.tobytes(order="C"))


def _science_artifact_fingerprint(stage, payload):
    """Fingerprint cached preprocessing products that feed a later fit."""
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": SCIENCE_ARTIFACT_SCHEMA_VERSION,
            "target_revision": SCIENCE_ARTIFACT_TARGET_REVISION,
            "stage": str(stage),
            "payload": payload,
        },
    )
    return hasher.hexdigest()


def _science_artifact_manifest_matches(path, fingerprint):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("schema_version") == SCIENCE_ARTIFACT_SCHEMA_VERSION
        and payload.get("target_revision") == SCIENCE_ARTIFACT_TARGET_REVISION
        and payload.get("fingerprint_sha256") == fingerprint
    )


def _write_science_artifact_manifest(path, stage, fingerprint):
    payload = {
        "schema_version": SCIENCE_ARTIFACT_SCHEMA_VERSION,
        "target_revision": SCIENCE_ARTIFACT_TARGET_REVISION,
        "stage": str(stage),
        "fingerprint_sha256": fingerprint,
    }
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _file_content_identity(path, block_size=8 * 1024 * 1024):
    """Return a stable source identity including a complete content digest."""
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    before = os.stat(resolved)
    digest = hashlib.sha256()
    with open(resolved, "rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    after = os.stat(resolved)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"Source file changed while hashing: {resolved}")
    return {
        "realpath": resolved,
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _optional_file_content_identity(path):
    if path is None:
        return None
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved):
        return {"realpath": resolved, "missing": True}
    return _file_content_identity(resolved)


def _directory_metadata_identity(path):
    """Fingerprint a model-grid tree without rereading multi-GB grid files."""
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved):
        return {"realpath": resolved, "missing": True}
    entries = []
    for root, dirs, files in os.walk(resolved):
        dirs.sort()
        for name in sorted(files):
            full_path = os.path.join(root, name)
            stat = os.stat(full_path)
            entries.append(
                (
                    os.path.relpath(full_path, resolved),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
            )
    hasher = hashlib.sha256()
    _update_checkpoint_hash(hasher, entries)
    return {
        "realpath": resolved,
        "file_count": len(entries),
        "metadata_sha256": hasher.hexdigest(),
    }


def _atomic_save_npy(path, values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npy"
    np.save(temporary, values)
    os.replace(temporary, path)


def _atomic_savez(path, **values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    np.savez(temporary, **values)
    os.replace(temporary, path)


def _atomic_savez_compressed(path, **values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _atomic_dataframe_csv(frame, path, *, index=False):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)
