"""Second, independent anonymity scan of an unpacked package: broad identifier patterns, not the terms.

    python tools/scan_identifiers.py <dir> --patterns .identifier-patterns.txt [--people <git repo>]...

The patterns file holds ``category<TAB>regex`` lines and ``allow<TAB>regex`` lines (a match that
fully matches an allow regex is public and not reported); ``--people`` adds every author and
committer name and e-mail of a repository's history as the category ``person``. Every file is read
the way anonymize.py reads it (text, database cells, PDF streams and text, PNG text chunks, archive
members, names), but with these patterns. Prints ``file<TAB>category<TAB>count`` per hit, never the
matched text, then a total; exit 1 on any hit.
"""

import argparse
import collections
import multiprocessing
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from anonymize import KEEP, kind, png_chunks, views  # noqa: E402

#: Tokens of a history that are no person (bots, placeholders).
NOT_PEOPLE = {"anonymous", "github", "claude", "noreply", "root", "users"}
PATTERNS: list[tuple[str, re.Pattern[str]]] = []
ALLOW: list[re.Pattern[str]] = []


def people(repos: list[pathlib.Path]) -> set[str]:
    """Every word (>= 4 letters) of the author and committer names and e-mail local parts of the repositories."""
    words = set()
    for repo in repos:
        log = subprocess.run(
            ["git", "-C", str(repo), "log", "--all", "--format=%an%n%cn%n%ae%n%ce"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in set(log.splitlines()):
            local = line.split("@")[0].split("+")[-1] if "@" in line else line
            words.update(w.lower() for w in WORD.findall(local) if len(w) >= 4)
    return words - NOT_PEOPLE


WORD = re.compile(r"[^\W\d_]+")
PEOPLE: set[str] = set()


def scan_one(item: tuple[pathlib.Path, pathlib.Path]) -> list[str]:
    path, root = item
    name = str(path.relative_to(root))
    counts: collections.Counter[str] = collections.Counter()
    for where, text in [(name, name), *views(name, path.read_bytes())]:
        for literal in KEEP:
            text = text.replace(literal, "\0")
        for category, pattern in PATTERNS:
            for match in pattern.finditer(text):
                if not any(allow.fullmatch(match.group(0)) for allow in ALLOW):
                    counts[category] += 1
        if PEOPLE:
            counts["person"] += sum(
                1
                for w in WORD.findall(text)
                if w.lower() in PEOPLE
                and not any(allow.fullmatch(w) for allow in ALLOW)
            )
    counts = +counts
    return [
        f"{name}\t{category}\t{count}" for category, count in sorted(counts.items())
    ]


def compressed_spans(path: pathlib.Path) -> list[tuple[int, int]]:
    """Byte ranges of compressed data (PNG pixels, PDF streams, whole archives) where a byte match is noise;
    the other two scans read these decompressed."""
    data = path.read_bytes()
    form = kind(path.name, data)
    if form == "archive":
        return [(0, len(data))]
    spans = []
    if form == "png":
        at = 8
        for chunk, body in png_chunks(data):
            if chunk == b"IDAT":
                spans.append((at + 8, at + 8 + len(body)))
            at += 12 + len(body)
    if form == "pdf":
        for match in re.finditer(rb"stream\r?\n", data):
            spans.append((match.end(), data.find(b"endstream", match.end())))
    return spans


def byte_scan(root: pathlib.Path) -> int:
    """``grep -a -i -o -b -r -P`` over every byte of the tree; a match inside compressed data is counted as noise."""
    words = "|".join(sorted(re.escape(w) for w in PEOPLE))
    pattern = "|".join(f"(?:{p.pattern})" for c, p in PATTERNS) + (
        f"|(?<![A-Za-z])(?:{words})(?![A-Za-z])" if words else ""
    )
    grep = subprocess.run(
        ["grep", "-r", "-a", "-i", "-o", "-b", "-P", "-e", pattern, str(root)],
        capture_output=True,
    )
    if grep.returncode > 1:
        raise RuntimeError(grep.stderr.decode())
    real, noise, spans = collections.Counter(), 0, {}
    for line in grep.stdout.decode("latin-1").splitlines():
        path, offset, text = line.split(":", 2)
        if any(text in literal for literal in KEEP) and any(
            literal.encode() in pathlib.Path(path).read_bytes() for literal in KEEP
        ):
            continue  # inside the one exempt literal (the file holds it; scans 1 and 2 check the rest)
        if any(allow.fullmatch(text) for allow in ALLOW):
            continue
        if path not in spans:
            spans[path] = compressed_spans(pathlib.Path(path))
        if any(a <= int(offset) < b for a, b in spans[path]):
            noise += 1
            continue
        category = next((c for c, p in PATTERNS if p.fullmatch(text)), "person")
        real[(str(pathlib.Path(path).relative_to(root)), category)] += 1
    for (path, category), count in sorted(real.items()):
        print(f"{path}\t{category}\t{count}")
    summary = ", ".join(
        f"{c}={n}"
        for c, n in sorted(collections.Counter(c for p, c in real.elements()).items())
    )
    print(
        f"{'clean' if not real else 'HITS'} (byte-level grep): {summary or 'none'}; {noise} match(es) inside compressed data"
    )
    return 1 if real else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--patterns", type=pathlib.Path, required=True)
    parser.add_argument("--people", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="literal exempt from the scan (anonymize.py --keep)",
    )
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="byte-level grep instead of the format-aware scan",
    )
    args = parser.parse_args()
    for line in args.patterns.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            category, regex = line.split("\t", 1)
            if category == "allow":
                ALLOW.append(re.compile(regex, re.IGNORECASE))
            else:
                PATTERNS.append((category, re.compile(regex, re.IGNORECASE)))
    PEOPLE.update(people(args.people))
    KEEP.extend(args.keep)
    if args.bytes:
        return byte_scan(args.root)
    items = sorted(
        ((p, args.root) for p in args.root.rglob("*") if p.is_file()),
        key=lambda i: -i[0].stat().st_size,
    )
    with multiprocessing.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
        hits = sorted(
            line for lines in pool.imap_unordered(scan_one, items) for line in lines
        )
    for line in hits:
        print(line)
    total = collections.Counter()
    for line in hits:
        total[line.split("\t")[1]] += int(line.split("\t")[2])
    summary = ", ".join(f"{c}={n}" for c, n in sorted(total.items())) or "none"
    print(
        f"{'clean' if not hits else 'HITS'}: {len(items)} files scanned, hits per category: {summary}"
    )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
