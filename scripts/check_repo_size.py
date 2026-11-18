"""Limit what reviewers must clone. No non-standard Python dependencies.

Default: measure the committed HEAD tree (also works with --depth 1).
--staged: measure the index, ignoring unstaged or untracked files.
--base REF: also check all blobs introduced in REF..HEAD, even when deleted
before HEAD. This mode requires complete commit/tree history; historical
blobs outside that range are not opened. Never traverse .git as a directory.
"""
import argparse
import json
import os
import subprocess
import sys


MIB = 1024 ** 2
MAX_FILE = 5 * MIB
MAX_NOTEBOOK_PAYLOAD = MIB
MAX_CHECKOUT = 24 * MIB   # Counts each path, including duplicate copies.
MAX_CHANGE = 32 * MIB     # Counts each newly reachable blob just once.


def git(*args, data=b""):
    return subprocess.run(
        ["git", *args], input=data, stdout=subprocess.PIPE, check=True
    ).stdout


def add(entries, oid, path, mode):
    # Gitlinks point into another repository, not a blob delivered in this one.
    if mode != b"160000":
        entries.setdefault(oid, set()).add((path, mode))


def snapshot(staged):
    entries = {}
    if staged:
        records = git("ls-files", "--stage", "--full-name", "-z")
    else:
        records = git("ls-tree", "--full-tree", "-rz", "HEAD")
    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        if staged:
            mode, oid, stage = metadata.split()
            if stage != b"0":
                raise ValueError(f"Unmerged index entry: {os.fsdecode(path)!r}")
        else:
            mode, _, oid = metadata.split()
        add(entries, oid, path, mode)
    return entries


def introduced(base):
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        raise ValueError("Range check needs full commit history; use a blob-filtered "
                         "clone without --depth, or omit --base for a snapshot check.")
    base_oid = git("rev-parse", "--verify", "--end-of-options",
                   base + "^{commit}").strip().decode("ascii")
    revision = base_oid + "..HEAD"
    new = set(git("rev-list", "--objects", "--no-object-names", revision).splitlines())
    entries = {}
    # NUL-separated raw diffs preserve spaces, tabs, newlines and hidden paths.
    # -m includes merge-resolution blobs; --no-renames keeps one path per pair.
    commits = git("rev-list", revision)
    fields = git("diff-tree", "--stdin", "--root", "-r", "-m", "--no-renames",
                 "--no-commit-id", "--raw", "--no-abbrev", "-z", data=commits).split(b"\0")
    for metadata, path in zip(fields[::2], fields[1::2]):
        _, mode, _, oid, _ = metadata.split()
        if oid in new:
            add(entries, oid, path, mode)
    return entries


def sizes(entries):
    output = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                 data=b"".join(k + b"\n" for k in entries))
    result = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[1] != b"blob":
            raise ValueError("Could not measure Git blob: " + line.decode("ascii", "replace"))
        result[fields[0]] = int(fields[2])
    return result


def notebook_payload(oid):
    # Count the serialized bytes committed as outputs/attachments, including
    # base64 and textual output. A minified encoding avoids counting indentation.
    doc = json.loads(git("cat-file", "blob", oid.decode("ascii")))
    cells = doc["cells"]
    if not isinstance(cells, list):
        raise ValueError("cells must be a list")
    return sum(len(json.dumps(cell[key], ensure_ascii=False, separators=(",", ":")).encode())
               for cell in cells for key in ("outputs", "attachments") if cell.get(key))


def check(staged=False, base=None):
    tree = snapshot(staged)
    history = introduced(base) if base is not None else {}
    entries = {oid: tree.get(oid, set()) | history.get(oid, set()) for oid in tree.keys() | history.keys()}
    blob_sizes = sizes(entries)
    problems = []
    for oid, paths in sorted(entries.items()):
        n = blob_sizes[oid]
        labels = ", ".join(repr(os.fsdecode(p)) for p, _ in sorted(paths))
        if n > MAX_FILE:
            problems.append(f"{labels}: blob is {n:,} bytes (limit {MAX_FILE:,})")
        elif any(p.lower().endswith(b".ipynb") and mode != b"120000" for p, mode in paths):
            try:
                embedded = notebook_payload(oid)
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                problems.append(f"{labels}: unreadable notebook: {exc}")
            else:
                if embedded > MAX_NOTEBOOK_PAYLOAD:
                    problems.append(f"{labels}: notebook outputs/attachments use {embedded:,} bytes "
                                    f"(limit {MAX_NOTEBOOK_PAYLOAD:,})")
    paths = sum(len(v) for v in tree.values())
    total = sum(len(v) * blob_sizes[k] for k, v in tree.items())
    print(f"{'Staged' if staged else 'HEAD'}: {paths} tracked files, {total:,} uncompressed bytes.")
    if total > MAX_CHECKOUT:
        problems.append(f"Checkout total is {total:,} bytes (limit {MAX_CHECKOUT:,})")
    if base is not None:
        total_new = sum(blob_sizes[k] for k in history)
        print(f"{base}..HEAD: {len(history)} new blobs, {total_new:,} uncompressed bytes.")
        if total_new > MAX_CHANGE:
            problems.append(f"New blob total is {total_new:,} bytes (limit {MAX_CHANGE:,})")
    for msg in problems:
        print(msg, file=sys.stderr)
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true", help="inspect files that would be committed")
    modes.add_argument("--base", metavar="REF", help="also check new history since REF")
    args = parser.parse_args()
    try:
        root = git("rev-parse", "--show-toplevel").rstrip(b"\n")
        os.chdir(os.fsdecode(root))
        return 0 if check(args.staged, args.base) else 1
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Size check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
