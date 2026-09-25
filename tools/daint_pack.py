"""Build the Daint regrade pack from local worklists: copy every named source under <out>/files and
rewrite paths to @PACK@/files/<path relative to the mirror>. Read-only on the mirror; creates <out>.

    python tools/daint_pack.py --mirror "$MIRROR" --out pack wl-llr.jsonl wl-llr-fortran.jsonl ...
"""

import argparse
import collections
import json
import pathlib
import shutil
from typing import Any

#: Arms whose .env file was never written; grading keys equal the model's base arm (checked 09-24).
ENV_FROM = {
    "gpu-llr-focus40-oss120b-hip-perf-playbook-amd": "gpu-llr-focus40-oss120b-hip",
    "gpu-llr-focus40-oss120b-hip-perf-playbook-amd-clean": "gpu-llr-focus40-oss120b-hip",
}
LANGUAGES = ["c", "fortran", "hip", "triton", "c-openmp"]


def backend(item: dict[str, Any]) -> str:
    """Language and backend of one worklist item: c, fortran, c-openmp, hip, triton."""
    if item["env"].get("HPCAGENT_BENCH_OFFLOAD"):
        return "c-openmp"
    return "triton" if item["language"] == "python" else str(item["language"])


def pack_sources(item: dict[str, Any], mirror: str, out: pathlib.Path) -> bool:
    """Copy the item's sources into the pack and rewrite their paths; False if one is missing."""
    fields = [field for field in ("source", "device_source") if item.get(field)]
    paths = {field: str(item[field]) for field in fields}
    # Check every source before copying any, so a half-packed item leaves nothing behind.
    if not all(path.startswith(mirror) and pathlib.Path(path).is_file() for path in paths.values()):
        return False
    for field, path in paths.items():
        rel = path.removeprefix(mirror)
        target = out / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        item[field] = "@PACK@/files/" + rel
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worklists", nargs="+", type=pathlib.Path)
    parser.add_argument("--mirror", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    args = parser.parse_args()
    mirror = str(args.mirror.resolve()) + "/"
    args.out.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = [
        json.loads(line) for path in args.worklists for line in path.read_text().splitlines() if line.strip()
    ]
    envs = {item["arm"]: item["env"] for item in items if item["env"]}
    missing_base = sorted({base for base in ENV_FROM.values() if base not in envs})
    seen: set[tuple[str, str, str, int]] = set()
    by_backend: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    missing: collections.Counter[str] = collections.Counter()
    for item in items:
        key = (item["db"], item["run_id"], item["benchmark"], item["ts_ms"])
        if key in seen:
            continue
        seen.add(key)
        if not item["env"] and item["arm"] in ENV_FROM:
            base = ENV_FROM[item["arm"]]
            if base in missing_base:
                parser.error(f"{item['arm']} borrows its env from {base}, which no worklist names")
            item["env"] = dict(envs[base])
        if backend(item) not in args.languages:
            continue
        if not pack_sources(item, mirror, args.out):
            missing[item["arm"]] += 1
            continue
        item["db"] = item["db"].removeprefix(mirror)
        by_backend[backend(item)].append(item)
    for name, rows in sorted(by_backend.items()):
        rows.sort(key=lambda row: (row["benchmark"], row["arm"], row["run_id"]))
        (args.out / f"wl-llr-{name}.portable.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        kernels, arms = len({row["benchmark"] for row in rows}), len({row["arm"] for row in rows})
        print(f"{name}: {len(rows)} items, {kernels} kernels, {arms} arms")
    if missing:
        print("source missing, not packed:", dict(missing))


if __name__ == "__main__":
    main()
