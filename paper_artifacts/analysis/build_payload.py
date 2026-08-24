"""Shape arms.json into exactly what the chart page needs, so the page holds no logic."""
import json
import math
import pathlib
import statistics

D = json.loads(pathlib.Path("arms.json").read_text())
PAIRS = [("qwen30b", "C", "605458", "605459"),
         ("oss120b", "C", "605460", "605461"),
         ("qwen30b", "Fortran", "605696", "605697")]


def sign_p(win: int, lose: int) -> float:
    n = win + lose
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(win, lose) + 1)) / 2**n)


ab, pooled = [], {"win": 0, "lose": 0, "tie": 0}
for model, lang, a, b in PAIRS:
    na, sk = D[a]["best"], D[b]["best"]
    both = sorted(set(na) & set(sk))
    pts = [{"k": k, "no": round(na[k], 4), "sk": round(sk[k], 4)} for k in both]
    win = sum(1 for p in pts if p["sk"] > p["no"] + 1e-9)
    lose = sum(1 for p in pts if p["sk"] < p["no"] - 1e-9)
    pooled["win"] += win
    pooled["lose"] += lose
    pooled["tie"] += len(pts) - win - lose
    ab.append({"model": model, "lang": lang, "points": pts, "win": win, "lose": lose,
               "tie": len(pts) - win - lose, "p": round(sign_p(win, lose), 4),
               "medNo": round(statistics.median(p["no"] for p in pts), 3),
               "medSk": round(statistics.median(p["sk"] for p in pts), 3)})
pooled["p"] = round(sign_p(pooled["win"], pooled["lose"]), 4)

arms = []
for job in sorted(D, key=lambda j: (D[j]["language"], D[j]["model"], D[j]["skills"])):
    d = D[job]
    st, tot = d["status"], max(1, sum(D[job]["status"].values()))
    vals = list(d["best"].values())
    arms.append({
        "job": d["job"], "model": d["model"], "lang": d["language"], "skills": d["skills"],
        "kernels": len(vals), "subs": d["submissions"],
        "median": round(statistics.median(vals), 3) if vals else None,
        "coarse": sum(1 for v in vals if v > 5.0),
        "status": {k: st.get(k, 0) for k in
                   ("ok", "incorrect", "build_error", "score_error", "timeout", "overfit")},
        "total": tot,
    })

pathlib.Path("payload.json").write_text(
    json.dumps({"ab": ab, "pooled": pooled, "arms": arms}, indent=1))
print("pooled:", pooled)
for a in ab:
    print(f"{a['model']:8s}/{a['lang']:8s} n={len(a['points']):3d} "
          f"+{a['win']} -{a['lose']} ={a['tie']} p={a['p']}")
print("arms:", len(arms))
