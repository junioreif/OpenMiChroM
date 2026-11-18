"""Exercise the release-size gate against real isolated Git histories."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_size.py"
SPEC = importlib.util.spec_from_file_location("check_repo_size", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def commit(message):
    git("add", "-A")
    git("commit", "-qm", message)
    return git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.chdir(root)
    git("init", "-q")
    for key, val in (("user.name", "Fixture"), ("user.email", "fixture@example.org"),
                     ("commit.gpgsign", "false"), ("core.hooksPath", ".git/unused-hooks")):
        git("config", key, val)
    (root / "code").write_text("start\n")
    commit("base")
    return root


def test_staged_reads_index_not_working_copy(repo, monkeypatch, capsys):
    monkeypatch.setattr(guard, "MAX_FILE", 12)
    f = repo / ".hidden name"
    f.write_bytes(b"x" * 13)
    assert guard.check(staged=True)  # Untracked, would not be sent.
    git("add", "--", f.name)
    f.write_bytes(b"small")
    assert not guard.check(staged=True)
    assert f.name in capsys.readouterr().err
    assert guard.check()  # HEAD has not changed.
    git("add", "--", f.name)
    assert guard.check(staged=True)


def test_deleted_intermediate_blob_still_counts(repo, monkeypatch, capsys):
    base = git("rev-parse", "HEAD")
    name = ".data\nwith tab\t"
    f = repo / name
    f.write_bytes(b"x" * 20)
    commit("tracking a result")
    f.unlink()
    commit("remove from checkout")
    monkeypatch.setattr(guard, "MAX_FILE", 19)
    assert guard.check()
    assert not guard.check(base=base)
    assert repr(name) in capsys.readouterr().err
    # The large blob is part of the base's *history*, not new in this range.
    assert guard.check(base=git("rev-parse", "HEAD^"))


def test_checkout_counts_duplicate_paths_range_counts_unique_blobs(repo, monkeypatch, capsys):
    base = git("rev-parse", "HEAD")
    for name in ["a", "b"]:
        (repo / name).write_bytes(b"x" * 10)
    commit("duplicate paths")
    monkeypatch.setattr(guard, "MAX_CHECKOUT", 25)
    monkeypatch.setattr(guard, "MAX_CHANGE", 10)
    assert not guard.check(base=base)
    out = capsys.readouterr()
    assert "Checkout total is 26" in out.err
    assert "1 new blobs, 10 uncompressed bytes" in out.out
    monkeypatch.setattr(guard, "MAX_CHECKOUT", 26)
    monkeypatch.setattr(guard, "MAX_CHANGE", 9)
    assert not guard.check(base=base)
    assert "New blob total is 10" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["outputs", "attachments"])
def test_detect_notebook_content_even_after_deletion(repo, monkeypatch, key, capsys):
    base = git("rev-parse", "HEAD")
    doc = {"cells": [{key: ["x"*256] if key == "outputs" else {"img": {"image/png": "x"*256}}}]}
    f = repo / "demo.ipynb"
    f.write_text(json.dumps(doc))
    commit("figure data")
    f.unlink()
    commit("remove notebook")
    monkeypatch.setattr(guard, "MAX_NOTEBOOK_PAYLOAD", 255)
    assert not guard.check(base=base)
    assert "notebook outputs/attachments" in capsys.readouterr().err


def test_shallow_snapshot_but_no_incomplete_history_claim(repo, tmp_path, monkeypatch):
    (repo / "code").write_text("updated\n")
    commit("update")
    clone = tmp_path / "shallow"
    git("clone", "-q", "--depth", "1", repo.as_uri(), str(clone))
    monkeypatch.chdir(clone)
    assert guard.check()
    with pytest.raises(ValueError, match="full commit history"):
        guard.check(base="HEAD")
