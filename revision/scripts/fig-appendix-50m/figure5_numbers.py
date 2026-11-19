"""Figure 5's numbers (no drawing: the bench has no plotting API for this figure) from the one-limit reruns: rho_C
per comparison and token weighting, with v1's computation (migrate-320a45c61/figure5.py ratios()): main's
paired_setups.tokens_by_setup_kernel priced by cost.priced under each card, and significance.paired.

    python figure5_numbers.py --harness20 obs-final-all.db --llr40 obs-main-0.db [...] --out figure5_numbers.csv
"""

import argparse
import math
import pathlib

import sys

import pandas as pd

import hpcagent_bench
from hpcagent_bench.stats import cost, significance

sys.path.insert(0, str(pathlib.Path(hpcagent_bench.__file__).parents[1] / "statistics"))
import paired_setups as ps  # noqa: E402

CARDS = (("effective", "1,0,1"), ("billed", "1,.1,1"), ("total", "1,1,1"))
#: (label, source, treated, control); v1's CPF row (cpf-llr-focus40-qwen38-c-cpfsrc-v2) has no rerun.
COMPARISONS = (
    ("OpenHands", "harness20", "harness20-qwen38-c-openhands", "harness20-qwen38-c"),
    ("HIP Skills", "llr40", "rev-llr40-oss120b-hip-lang-skills", "rev-llr40-oss120b-hip"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness20", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--llr40", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    sources = {"harness20": args.harness20, "llr40": args.llr40}
    rows = []
    for label, source, treated, control in COMPARISONS:
        frame = ps.load_observations(sources[source], cost.resolve("effective"))
        frame = frame[frame.setup.isin([treated, control])]
        for card, weights in CARDS:
            tokens = ps.tokens_by_setup_kernel(cost.priced(frame, cost.resolve(card)))
            kernels = sorted(k for (s, k) in tokens if s == treated and tokens.get((control, k), 0) > 0)
            logs = [math.log(tokens[(control, k)] / tokens[(treated, k)]) for k in kernels]
            test = significance.paired(logs)
            rows.append({"comparison": label, "treated": treated, "control": control, "card": card,
                         "weights": weights, "n": len(logs), "rho_c": math.exp(test.estimate),
                         "ci_low": math.exp(test.low), "ci_high": math.exp(test.high), "p": test.pvalue})
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    spread = table.groupby("comparison").rho_c.agg(lambda r: r.max() / r.min())
    print("weighting factor (max / min rho_C) per comparison:", spread.round(2).to_dict())


if __name__ == "__main__":
    main()
