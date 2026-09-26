"""Write anonymized copies of SQLite databases and text files for the anonymous release; the sources stay untouched.

    python tools/anonymize.py --out /path/to/anon [--commits <git repo>]... <dir>...
    python tools/anonymize.py --check [--commits <git repo>]... <dir>...

Every file under the given roots is copied: a ``*.db`` as a database, any other file as UTF-8 text,
and a binary unchanged, to ``<out>/<relative path>`` with the terms applied to every path component; every text value is
rewritten with the terms of ``.anonymize-terms.txt`` (``pattern=>replacement``; a bare pattern becomes
``XXXX``), and a database copy is vacuumed so no old page keeps the original text. A commit id of a
``--commits`` repository (7 to 40 hex digits) becomes a keyed pseudonym of the same length, so equal
ids stay equal but none can be looked up in a public repository. The run fails if any term
still matches a copy: its raw bytes, a PNG's text chunks, a PDF's inflated streams and text, or any
member of an archive. An archive is unpacked, rewritten member by member and repacked; a PNG's text
chunks and a PDF (through qpdf) are rewritten only when a term matches them, so clean figures stay
byte-identical; any other binary type stops the run. Reports name a term by its line in the terms
file, never by its text.
"""

import argparse
import bz2
import gzip
import hashlib
import hmac
import io
import lzma
import multiprocessing
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterator

DEFAULT_REPLACEMENT = "XXXX"
#: Terms made of letters only match between non-letters, so "cat" does not hit "concat" but does hit
#: "submit_cat.sh" or "cat2" (``\b`` would treat "_" and digits as part of the word).
WORD_TERM = re.compile(r"[A-Za-z\[\] ]+")
#: Account names such as "g34" must not match inside other tokens or record bytes ("Bg341.3").
ACCOUNT_TERM = re.compile(r"[A-Za-z-]+[0-9]+")
HEX_RUN = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{7,40}(?![0-9A-Fa-f])")
ARCHIVES = (
    ".zip",
    ".npz",
    ".whl",
    ".tar",
    ".tgz",
    ".gz",
    ".zst",
    ".tzst",
    ".bz2",
    ".xz",
)

#: (pattern, replacement, label); the label ("term 12") names the line, never the text.
Term = tuple[re.Pattern[str], str | Callable[[re.Match[str]], str], str]


#: Lower-case text every match of a term contains: a value without it needs no regex pass.
LITERAL: dict[str, str] = {}
ESCAPED = re.compile(r"\(\?:((?:\\.|[^\\()\[\]{}.*+?^$|])+)\)")


def required_literal(pattern: str) -> str:
    """The literal a plain word, account or ``(?:escaped literal)`` term must contain; empty for any other regex."""
    if (
        WORD_TERM.fullmatch(pattern)
        and not set("[]") & set(pattern)
        or ACCOUNT_TERM.fullmatch(pattern)
    ):
        return pattern.lower()
    escaped = ESCAPED.fullmatch(pattern)
    return re.sub(r"\\(.)", r"\1", escaped.group(1)).lower() if escaped else ""


def load_terms(path: pathlib.Path) -> list[Term]:
    """Compiled (pattern, replacement, label) triples, case-insensitive, in file order."""
    terms = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        pattern, arrow, replacement = line.partition("=>")
        literal = required_literal(pattern)
        if literal:
            LITERAL[f"term {number}"] = literal
        if WORD_TERM.fullmatch(pattern):
            pattern = rf"(?<![A-Za-z]){pattern}(?![A-Za-z])"
        elif ACCOUNT_TERM.fullmatch(pattern):
            pattern = rf"(?<![A-Za-z0-9]){pattern}(?![0-9])"
        terms.append(
            (
                re.compile(pattern, re.IGNORECASE),
                replacement or DEFAULT_REPLACEMENT,
                f"term {number}",
            )
        )
    return terms


def commit_term(repos: list[pathlib.Path], key: bytes) -> Term:
    """One term that maps every prefix (>= 7 digits) of a commit id of ``repos`` to a keyed pseudonym."""
    commits = set()
    for repo in repos:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-list", "--all"],
            capture_output=True,
            text=True,
            check=True,
        )
        commits.update(out.stdout.split())
    by_prefix: dict[str, list[str]] = {}
    for commit in commits:
        by_prefix.setdefault(commit[:7], []).append(commit)

    def known(token: str) -> bool:
        return any(c.startswith(token) for c in by_prefix.get(token[:7], ()))

    def pseudonym(match: re.Match[str]) -> str:
        token = match.group(0)
        if not known(token):
            return token
        salt = 0
        while True:
            fake = hmac.new(
                key, f"{token}:{salt}".encode(), hashlib.sha256
            ).hexdigest()[: len(token)]
            if not known(fake):
                return fake
            salt += 1

    return HEX_RUN, pseudonym, f"commit ids of {len(commits)} commits"


#: Literals no term rewrites and no check reports (``--keep``), e.g. a dependency URL that must still install.
KEEP: list[str] = []


def scrub(text: str, terms: list[Term]) -> str:
    """Apply every term to one value, leaving every ``KEEP`` literal as it is."""
    if not isinstance(text, str):
        return text
    for literal in KEEP:
        if literal in text:
            return literal.join(scrub(part, terms) for part in text.split(literal))
    for pattern, replacement, label in terms:
        if label not in LITERAL or LITERAL[label] in text.lower():
            text = pattern.sub(replacement, text)
    return text


def anonymize(source: pathlib.Path, target: pathlib.Path, terms: list[Term]) -> None:
    """Copy one database to target and rewrite every text value in the copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)
    connection = sqlite3.connect(target)
    seen: dict[
        str, str
    ] = {}  # cells repeat (paths, models, nodes): scrub each value once

    def anon(value: str) -> str:
        if value not in seen:
            seen[value] = scrub(value, terms)
        return seen[value]

    connection.create_function("anon", 1, anon, deterministic=True)
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]
    for table in tables:
        columns = [
            c[1] for c in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        ]
        if columns:
            connection.execute(
                f'UPDATE "{table}" SET '
                + ", ".join(f'"{c}" = anon("{c}")' for c in columns)
            )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


SQLITE = b"SQLite format 3\0"
#: Binaries that carry no text of their own beyond what the raw-byte check reads: they are copied, then checked.
OPAQUE = {
    b"\x7fELF": "ELF object",
    b"\x93NUMPY": "numpy array",
    b"\xff\xd8\xff": "JPEG",
    b"GIF8": "GIF",
}
#: Stored figures and data are left byte-identical unless a term matches them.
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def kind(name: str, data: bytes) -> str:
    """text, sqlite, png, pdf, archive or opaque; any other binary raises, so nothing is copied unchecked."""
    if data[:16] == SQLITE:
        return "sqlite"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:5] == b"%PDF-":
        return "pdf"
    if (
        name.lower().endswith(ARCHIVES)
        or data[:4] in (b"PK\x03\x04", b"\x28\xb5\x2f\xfd")
        or data[:2] == b"\x1f\x8b"
    ):
        return "archive"
    if any(data.startswith(magic) for magic in OPAQUE):
        return "opaque"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"unknown binary type, not copied unchecked: {name} ({data[:8]!r})"
        ) from None
    return "text"


def png_chunks(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """(kind, body) of every chunk."""
    at = 8
    while at + 8 <= len(data):
        size = int.from_bytes(data[at : at + 4], "big")
        yield data[at + 4 : at + 8], data[at + 8 : at + 8 + size]
        at += 12 + size


def png_text(kind_: bytes, body: bytes) -> bytes:
    """A text chunk's key and text, inflated."""
    if kind_ == b"zTXt":
        key, nul, rest = body.partition(b"\0")
        return key + b"\n" + zlib.decompress(rest[1:])
    if kind_ == b"iTXt":
        key, flag, rest = body.partition(b"\0")
        compressed, rest = rest[:1], rest[2:]
        language, nul, rest = rest.partition(b"\0")
        translated, nul, text = rest.partition(b"\0")
        return b"\n".join(
            (
                key,
                language,
                translated,
                zlib.decompress(text) if compressed == b"\1" else text,
            )
        )
    return body


def rewrite_png(data: bytes, terms: list[Term]) -> bytes:
    """The PNG with every text chunk (tEXt, zTXt, iTXt) rewritten as a tEXt; pixels untouched."""
    out = [data[:8]]
    for kind_, body in png_chunks(data):
        if kind_ in (b"tEXt", b"zTXt", b"iTXt"):
            key, nul, text = png_text(kind_, body).partition(b"\n")
            if kind_ == b"iTXt":
                text = text.split(b"\n")[-1]
            kind_ = b"tEXt"
            body = (
                scrub(key.decode("latin-1"), terms).encode("latin-1")
                + b"\0"
                + scrub(text.decode("latin-1"), terms).encode("latin-1", "replace")
            )
        out += [
            len(body).to_bytes(4, "big"),
            kind_,
            body,
            zlib.crc32(kind_ + body).to_bytes(4, "big"),
        ]
    return b"".join(out)


def pdf_streams(data: bytes) -> bytes:
    """Every Flate stream of a PDF inflated (content, metadata, XMP)."""
    parts = []
    for match in re.finditer(rb"stream\r?\n", data):
        try:
            parts.append(
                zlib.decompressobj().decompress(
                    data[match.end() : data.find(b"endstream", match.end())]
                )
            )
        except zlib.error:
            continue
    return b"\n".join(parts)


def pdf_text(data: bytes) -> bytes:
    """The text a reader sees (pdftotext), empty when poppler is missing."""
    if not shutil.which("pdftotext"):
        return b""
    return subprocess.run(
        ["pdftotext", "-", "-"], input=data, capture_output=True, check=True
    ).stdout


def rewrite_pdf(data: bytes, terms: list[Term]) -> bytes:
    """The PDF uncompressed (qpdf QDF), every term applied, offsets repaired and recompressed."""
    qdf = subprocess.run(
        ["qpdf", "--qdf", "--object-streams=disable", "-", "-"],
        input=data,
        capture_output=True,
        check=True,
    ).stdout
    qdf = scrub(qdf.decode("latin-1"), terms).encode("latin-1", "replace")
    fixed = subprocess.run(
        ["fix-qdf"], input=qdf, capture_output=True, check=True
    ).stdout
    return subprocess.run(
        ["qpdf", "--compress-streams=y", "--object-streams=generate", "-", "-"],
        input=fixed,
        capture_output=True,
        check=True,
    ).stdout


def unpack(name: str, data: bytes) -> tuple[str, list[tuple[str, bytes]]]:
    """(format, [(name, bytes)]) of an archive; a single compressed file is its own member."""
    low = name.lower()
    if data[:4] == b"PK\x03\x04" or low.endswith((".zip", ".npz", ".whl")):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return "zip", [
                (i.filename, archive.read(i))
                for i in archive.infolist()
                if not i.is_dir()
            ]
    codec = ""
    if data[:4] == b"\x28\xb5\x2f\xfd":
        codec, data = (
            "zst",
            subprocess.run(
                ["zstd", "-dc"], input=data, capture_output=True, check=True
            ).stdout,
        )
    elif data[:2] == b"\x1f\x8b":
        codec, data = "gz", gzip.decompress(data)
    elif data[:3] == b"BZh":
        codec, data = "bz2", bz2.decompress(data)
    elif data[:6] == b"\xfd7zXZ\0":
        codec, data = "xz", lzma.decompress(data)
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            return f"tar.{codec}", [
                (m.name, archive.extractfile(m).read())
                for m in archive.getmembers()
                if m.isfile()
            ]
    except tarfile.TarError:
        return codec, [(re.sub(r"\.(zst|tzst|gz|tgz|bz2|xz)$", "", name), data)]


def pack(form: str, files: list[tuple[str, bytes]]) -> bytes:
    """An archive of ``files``, reproducible: fixed times, no owner names."""
    buffer = io.BytesIO()
    if form == "zip":
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, body in files:
                archive.writestr(zipfile.ZipInfo(name, FIXED_ZIP_TIME), body)
        return buffer.getvalue()
    codec = form.removeprefix("tar.") if form.startswith("tar.") else form
    if form.startswith("tar."):
        with tarfile.open(
            fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for name, body in sorted(files):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(body), 0o644, 0
                archive.addfile(info, io.BytesIO(body))
        data = buffer.getvalue()
    else:
        data = files[0][1]
    if codec == "zst":
        return subprocess.run(
            ["zstd", "-q", "-19", "-c"], input=data, capture_output=True, check=True
        ).stdout
    return {
        "gz": lambda d: gzip.compress(d, mtime=0),
        "bz2": bz2.compress,
        "xz": lzma.compress,
    }.get(codec, bytes)(data)


def rewrite(name: str, data: bytes, terms: list[Term]) -> bytes:
    """The anonymized bytes of one file of any known type; a binary is rewritten only when a term matches it."""
    form = kind(name, data)
    if form == "text":
        return scrub(data.decode("utf-8"), terms).encode("utf-8")
    if form == "sqlite":
        with tempfile.TemporaryDirectory() as tmp:
            source, target = pathlib.Path(tmp, "in.db"), pathlib.Path(tmp, "out.db")
            source.write_bytes(data)
            anonymize(source, target, terms)
            return target.read_bytes()
    if form == "archive":
        form_, files = unpack(name, data)
        new = [
            (scrub(member, terms), rewrite(f"{name}!{member}", body, terms))
            for member, body in files
        ]
        return data if new == files else pack(form_, new)
    if not any(
        matches(term, text) for where, text in views(name, data) for term in terms
    ):
        return data
    if form == "png":
        return rewrite_png(data, terms)
    if form == "pdf":
        return rewrite_pdf(data, terms)
    return data  # opaque: the check below fails the release


def views(name: str, data: bytes) -> Iterator[tuple[str, str]]:
    """(where, text) a leak check reads: the raw bytes, a PNG's non-pixel chunks, a PDF's inflated streams
    and visible text, every member of an archive with its name (recursively), every cell of a database."""
    form = kind(name, data)
    if form == "png":
        yield (
            name,
            b"\n".join(
                k + b"\n" + png_text(k, b) for k, b in png_chunks(data) if k != b"IDAT"
            ).decode("latin-1"),
        )
        return
    if form == "archive":
        form_, files = unpack(name, data)
        if form_ == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                yield f"{name}[comment]", archive.comment.decode("latin-1")
        for member, body in files:
            yield f"{name}!{member}[name]", member
            yield from views(f"{name}!{member}", body)
        return
    if form == "sqlite":
        yield from cells(name, data)
        return
    if data.startswith(b"\x93NUMPY"):
        size = (
            10 + int.from_bytes(data[8:10], "little")
            if data[6] == 1
            else 12 + int.from_bytes(data[8:12], "little")
        )
        if re.search(rb"'descr': '[<>|=]?[fiucb?]", data[:size]):
            yield (
                f"{name}[header]",
                data[:size].decode("latin-1"),
            )  # numbers only after the header
            return
    yield name, data.decode("latin-1")
    if form == "pdf":
        yield f"{name}[streams]", pdf_streams(data).decode("latin-1")
        yield f"{name}[text]", pdf_text(data).decode("utf-8", "replace")


def cells(name: str, data: bytes) -> Iterator[tuple[str, str]]:
    """Every text or blob column of a database, its cells joined by newlines (raw bytes run adjacent cells together),
    and the raw bytes too when free pages could still hold old text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "check.db")
        path.write_bytes(data)
        connection = sqlite3.connect(path)
        if connection.execute("PRAGMA freelist_count").fetchone()[0]:
            yield f"{name}[raw bytes: free pages]", data.decode("latin-1")
        for table, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall():
            yield f"{name}[schema {table}]", sql
        for (table,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            for column in [
                c[1] for c in connection.execute(f'PRAGMA table_info("{table}")')
            ]:
                values = connection.execute(
                    f'SELECT "{column}" FROM "{table}" WHERE typeof("{column}") IN (\'text\', \'blob\')'
                )
                text = "\n".join(
                    v.decode("latin-1") if isinstance(v, bytes) else v
                    for (v,) in values
                )
                yield f"{name}[{table}.{column}]", text
        connection.close()


def matches(term: Term, text: str) -> bool:
    """Whether a term still finds something to replace in ``text``."""
    pattern, replacement, label = term
    for literal in KEEP:
        text = text.replace(literal, "\0")
    if callable(replacement):
        return any(replacement(m) != m.group(0) for m in pattern.finditer(text))
    return pattern.search(text) is not None


def leaks(target: pathlib.Path, terms: list[Term], name: str) -> list[str]:
    """``where:term N`` for every term that still matches a copy or its path."""
    found = []
    for where, text in [(f"{name}[path]", name), *views(name, target.read_bytes())]:
        for literal in KEEP:
            text = text.replace(literal, "\0")
        low = text.lower()
        found += [
            f"{where}:{t[2]}"
            for t in terms
            if (t[2] not in LITERAL or LITERAL[t[2]] in low) and matches(t, text)
        ]
    return found


#: The terms of this run, inherited by the forked workers (a commit term's closure does not pickle).
TERMS: list[Term] = []


def parallel(
    job: Callable[[tuple[pathlib.Path, ...]], str],
    items: list[tuple[pathlib.Path, ...]],
) -> Iterator[str]:
    """``job`` over ``items`` on forked workers, largest file first so the big databases start at once."""
    items = sorted(items, key=lambda item: -item[0].stat().st_size)
    with multiprocessing.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
        yield from pool.imap_unordered(job, items)


def check_one(item: tuple[pathlib.Path, ...]) -> str:
    path, root = item
    hits = leaks(path, TERMS, str(path.relative_to(root)))
    return f"LEAK\t{path}\t{' '.join(hits)}" if hits else ""


def check(roots: list[pathlib.Path], terms: list[Term]) -> int:
    """Report every file under ``roots`` a term still matches (bytes, members or path); 1 if any does."""
    TERMS[:] = terms
    items = [(p, root) for root in roots for p in root.rglob("*") if p.is_file()]
    found = sorted(line for line in parallel(check_one, items) if line)
    for line in found:
        print(line)
    print(
        f"{'clean' if not found else f'{len(found)} file(s) leak'}: {len(items)} files checked"
    )
    return 1 if found else 0


def anonymize_one(item: tuple[pathlib.Path, ...]) -> str:
    source, target, out = item
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.read_bytes()[:16] == SQLITE:
        anonymize(source, target, TERMS)
    else:
        target.write_bytes(rewrite(str(source), source.read_bytes(), TERMS))
    shutil.copymode(source, target)
    found = leaks(target, TERMS, str(target.relative_to(out)))
    return f"{'LEAK' if found else 'ok'}\t{target}\t{' '.join(found)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="+",
        type=pathlib.Path,
        help="directories whose files are anonymized",
    )
    parser.add_argument("--out", type=pathlib.Path, help="directory for the copies")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; report every file a term still matches",
    )
    parser.add_argument(
        "--terms", type=pathlib.Path, default=pathlib.Path(".anonymize-terms.txt")
    )
    parser.add_argument(
        "--commits",
        type=pathlib.Path,
        action="append",
        default=[],
        help="git repository whose commit ids are pseudonymized (repeatable)",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="literal left as it is and never reported",
    )
    args = parser.parse_args()
    KEEP.extend(args.keep)
    terms = load_terms(args.terms)
    if args.commits:
        terms.append(
            commit_term(args.commits, hashlib.sha256(args.terms.read_bytes()).digest())
        )
    if args.check:
        return check(args.roots, terms)
    if args.out is None:
        parser.error("--out is required unless --check")
    out = args.out.resolve()
    TERMS[:] = terms
    items = []
    for root in args.roots:
        for source in sorted(root.rglob("*")):
            source = source.resolve()
            if (
                out in source.parents
                or not source.is_file()
                or source.suffix in (".db-shm", ".db-wal")
            ):
                continue
            relative = source.relative_to(pathlib.Path.cwd().resolve())
            items.append(
                (
                    source,
                    out
                    / pathlib.Path(*(scrub(part, terms) for part in relative.parts)),
                    out,
                )
            )
    failed = False
    for line in parallel(anonymize_one, items):
        failed |= line.startswith("LEAK")
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
