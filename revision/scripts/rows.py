"""Table 18 rows from paired_setups.py CSVs: Solved control/treated, dR, rho_S [CI], rho_C [CI], R/S/C, MDE_S, MDE_C."""
import csv
import sys

LETTER = {"different": "d", "equivalent": "e", "inconclusive": "i"}


def fmt(x: str) -> str:
    return f"{float(x):.2f}"


def rows(path: str) -> list[str]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_pair.setdefault((r["setup_a"], r["setup_b"]), {})[r["leg"]] = r
    out = []
    for (a, b), legs in by_pair.items():
        sp, tk = legs["speedup"], legs["tokens"]
        s = sp
        n = int(s["n_roster"])
        dr = 100.0 * float(s["rate_difference"])
        drs = "0" if dr == 0 else f"{dr:+.1f}"
        verd = "/".join(x[0] for x in (sp["coverage_conclusion"], sp["conclusion"], tk["conclusion"]))
        out.append(
            f"{a} vs {b}: n={n} | {s['n_b']}/{s['n_a']} | {drs} | {fmt(sp['rho'])} [{fmt(sp['ci_low'])}, {fmt(sp['ci_high'])}]"
            f" | {fmt(tk['rho'])} [{fmt(tk['ci_low'])}, {fmt(tk['ci_high'])}] | {verd} | {fmt(sp['mde_ratio'])} | {fmt(tk['mde_ratio'])}"
            f" | limits {sp['limits_match']}")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("==", p)
        print("\n".join(rows(p)))
