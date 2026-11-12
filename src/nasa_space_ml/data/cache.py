"""Content-addressed local cache for NASA snapshots.

Why this exists: a capstone whose numbers change every time you re-run it is not
reproducible. Every download is recorded with a sha256, so `train` reads exactly the bytes
`fetch` wrote, and any hand-edit of a data file is detected instead of silently absorbed.

Writes are atomic (temp file + os.replace) so an interrupted download can never leave a
half-written snapshot that the next stage would happily train on.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..errors import DataError

MANIFEST_NAME = "checksum.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Cache:
    """A directory of snapshot files plus a manifest describing them."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- raw access

    def path(self, name: str) -> Path:
        return self.directory / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write_text(self, name: str, text: str) -> Path:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_bytes(self, name: str, payload: bytes) -> Path:
        target = self.path(name)
        handle = tempfile.NamedTemporaryFile(dir=self.directory, delete=False)  # noqa: SIM115
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, target)
        self._record(name, payload)
        return target

    def read_bytes(self, name: str) -> bytes:
        target = self.path(name)
        if not target.exists():
            raise DataError(
                f"cached artifact {name!r} not found; run `nasa-space-ml fetch` first",
                details={"path": str(target)},
            )
        payload = target.read_bytes()
        if not self.verify(name):
            raise DataError(
                f"cached artifact {name!r} does not match its recorded checksum; "
                "delete it and re-run `nasa-space-ml fetch`",
                details={"path": str(target)},
            )
        return payload

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")

    # ---------------------------------------------------------------- manifest

    @property
    def manifest_path(self) -> Path:
        return self.path(MANIFEST_NAME)

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _record(self, name: str, payload: bytes) -> None:
        entries = self.manifest()
        entries[name] = {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True))

    def verify(self, name: str) -> bool:
        """True when the file on disk matches the manifest (or is not listed in it)."""
        entry = self.manifest().get(name)
        if entry is None:
            return True
        if not self.path(name).exists():
            return False
        return sha256_bytes(self.path(name).read_bytes()) == entry["sha256"]

    def dataset_fingerprint(self) -> str:
        """A single hash over every recorded artifact, for model metadata provenance."""
        entries = self.manifest()
        digest = hashlib.sha256()
        for name in sorted(entries):
            digest.update(name.encode("utf-8"))
            digest.update(entries[name]["sha256"].encode("utf-8"))
        return digest.hexdigest()
