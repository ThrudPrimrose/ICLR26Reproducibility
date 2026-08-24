"""Per-kernel memory footprint of one grade, at each size preset, for the focus40 tag.

A submit holds ~7 copies of the array set live at once (public data, fresh-seed data, both
numpy references, o1 and o2 for determinism, c_public for the dual-oracle), so the column
that matters operationally is bytes x 7 x the judge node's grade concurrency.
"""
import pathlib
import re
import sys

BENCH = pathlib.Path("/capstor/scratch/cscs/ybudanaz/x86_64/optarena/hpcagent_bench/benchmarks")
COPIES = 7
ELEM = 8  # float64


def parse(yaml_text: str):
    """-> ({preset: {symbol: value}}, [(array, [dims...])])"""
    presets, arrays = {}, []
    cur = None
    in_arrays = False
    for line in yaml_text.splitlines():
        m = re.match(r"^\s{2}([A-Z]+):\s*$", line)
        if m:
            cur = m.group(1)
            presets[cur] = {}
            in_arrays = False
            continue
        m = re.match(r"^\s{4}(\w+):\s*(\d+)\s*$", line)
        if m and cur:
            presets[cur][m.group(1)] = int(m.group(2))
            continue
        if re.match(r"^\s{2}arrays:\s*$", line):
            in_arrays = True
            cur = None
            continue
        m = re.match(r"^\s{4}(\w+):\s*\(([^)]*)\)", line)
        if m and in_arrays:
            arrays.append((m.group(1), [d.strip() for d in m.group(2).split(",") if d.strip()]))
    return presets, arrays


def elems(dims, syms):
    n = 1
    for d in dims:
        if d.isdigit():
            n *= int(d)
        elif d in syms:
            n *= syms[d]
        else:
            return None
    return n


def main() -> None:
    keys = pathlib.Path(sys.argv[1]).read_text().strip().split(",")
    rows = []
    for key in keys:
        stem = key.split("/")[-1]
        hits = list(BENCH.glob(f"*/{stem}/{stem}.yaml")) or list(BENCH.glob(f"*/*/{stem}/{stem}.yaml"))
        if not hits:
            continue
        presets, arrays = parse(hits[0].read_text())
        out = {}
        for p, syms in presets.items():
            total = 0
            for _name, dims in arrays:
                n = elems(dims, syms)
                if n is None:
                    total = None
                    break
                total += n * ELEM
            out[p] = total
        rows.append((stem, out))
    rows.sort(key=lambda r: -(r[1].get("XL") or 0))
    print(f"{'kernel':30s} {'S':>9s} {'M':>9s} {'L':>9s} {'XL':>9s}   {'submit peak @XL':>15s}")
    grand = 0
    for stem, out in rows:
        cells = "".join(f"{(out.get(p) or 0)/2**30:8.2f}G" for p in ("S", "M", "L", "XL"))
        xl = out.get("XL") or 0
        grand += xl
        flag = "  <-- oversized" if xl * COPIES / 2**30 > 10 else ""
        print(f"{stem:30s} {cells}   {xl*COPIES/2**30:13.1f}G{flag}")
    print(f"\nkernels: {len(rows)}   total XL bytes across the tag: {grand/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
