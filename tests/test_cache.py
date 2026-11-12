"""The snapshot cache: content-addressed, atomic, and paranoid on read.

Why a cache at all: the two NASA pulls total tens of megabytes and take minutes, and a
capstone reviewer cannot re-download DR25 to check a number. Why checksums: a truncated
CSV from an interrupted download otherwise parses fine and silently trains on a subset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nasa_space_ml.data.cache import MANIFEST_NAME, Cache, sha256_bytes
from nasa_space_ml.errors import DataError

PAYLOAD = "a,b\n1,x\n2,y\n"


def test_sha256_bytes_is_stable_and_hex() -> None:
    digest = sha256_bytes(b"hello")
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert len(digest) == 64
    assert sha256_bytes(b"hello") == digest  # pure function, no hidden state


def test_directory_is_created_eagerly(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "down"
    Cache(nested)
    assert nested.is_dir()


def test_roundtrip_text_and_bytes(cache: Cache) -> None:
    cache.write_text("one.csv", PAYLOAD)
    assert cache.read_text("one.csv") == PAYLOAD
    cache.write_bytes("two.bin", b"\x00\x01\x02")
    assert cache.read_bytes("two.bin") == b"\x00\x01\x02"


def test_missing_artifact_tells_you_the_command_to_run(cache: Cache) -> None:
    with pytest.raises(DataError, match="run `nasa-space-ml fetch`"):
        cache.read_bytes("absent.csv")


def test_manifest_records_checksum_and_size(cache: Cache) -> None:
    cache.write_text("one.csv", PAYLOAD)
    entry = cache.manifest()["one.csv"]
    assert entry["sha256"] == sha256_bytes(PAYLOAD.encode())
    assert entry["bytes"] == len(PAYLOAD.encode())
    assert entry["written_utc"].endswith("+00:00")


def test_tampered_file_is_rejected_on_read(cache: Cache) -> None:
    """The single most important cache test: silent corruption must become a loud error."""
    cache.write_text("one.csv", PAYLOAD)
    cache.path("one.csv").write_text("a,b\n1,x\n", encoding="utf-8")
    assert cache.verify("one.csv") is False
    with pytest.raises(DataError, match="does not match its recorded checksum"):
        cache.read_text("one.csv")


def test_unlisted_file_verifies_true(cache: Cache) -> None:
    """A file dropped in by hand is not corrupt, it is simply outside the manifest."""
    cache.path("hand.csv").write_text("x", encoding="utf-8")
    assert cache.verify("hand.csv") is True


def test_manifest_entry_without_a_file_fails_verification(cache: Cache) -> None:
    cache.write_text("one.csv", PAYLOAD)
    cache.path("one.csv").unlink()
    assert cache.exists("one.csv") is False
    assert cache.verify("one.csv") is False


def test_write_is_atomic_so_a_crash_leaves_no_partial_file(cache: Cache, tmp_path: Path) -> None:
    """`os.replace` is the guarantee: the target either has the old bytes or the new ones.
    A leftover NamedTemporaryFile is noise; a half-written CSV is a wrong model."""
    cache.write_text("one.csv", PAYLOAD)
    leftovers = [p.name for p in tmp_path.rglob("*") if p.name.startswith("tmp")]
    assert not [p for p in leftovers if p.endswith(".csv")]
    assert cache.read_text("one.csv") == PAYLOAD


def test_second_write_replaces_content_and_updates_the_manifest(cache: Cache) -> None:
    cache.write_text("one.csv", "first\n")
    cache.write_text("one.csv", "second\n")
    assert cache.read_text("one.csv") == "second\n"
    assert cache.verify("one.csv")


def test_fingerprint_changes_when_any_artifact_changes(cache: Cache) -> None:
    cache.write_text("one.csv", PAYLOAD)
    before = cache.dataset_fingerprint()
    cache.write_text("two.csv", PAYLOAD)
    assert cache.dataset_fingerprint() != before
    assert before != ""


def test_fingerprint_is_independent_of_write_order(tmp_path: Path) -> None:
    first, second = Cache(tmp_path / "a"), Cache(tmp_path / "b")
    first.write_text("x.csv", "1")
    first.write_text("y.csv", "2")
    second.write_text("y.csv", "2")
    second.write_text("x.csv", "1")
    assert first.dataset_fingerprint() == second.dataset_fingerprint()


def test_manifest_is_json_and_sorted(cache: Cache) -> None:
    cache.write_text("z.csv", "1")
    cache.write_text("a.csv", "2")
    names = list(json.loads(cache.manifest_path.read_text(encoding="utf-8")))
    assert names == ["a.csv", "z.csv"]
    assert cache.manifest_path.name == MANIFEST_NAME
