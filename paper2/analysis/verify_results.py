"""Recomputes every number of the paper from the logs: per-seed counts, means, and the exact McNemar tests.
Usage: python verify_results.py   (run from this folder; needs numpy and scipy)"""
import numpy as np
from scipy.stats import binomtest
H = [l.split("|") for l in open("../data/facts_150.txt", encoding="utf8") if l.strip() and not l.startswith("#")]
CRIT = [i for i, h in enumerate(H) if h[1] == "1"]
KEYS = "criticos triviales trampas explicitos implicitos es en cultura_antes cultura salientes_criticos".split()

def read(files, after=None):
    R = {}
    for f in files:
        t = open(f, encoding="utf8").read()
        if after: t = t.split(after)[-1]
        for l in t.splitlines():
            p = l.strip().split(",")
            if len(p) == 13 and len(p[12]) == 150:
                R[(p[0], int(p[1]))] = dict(zip(KEYS, map(int, p[2:12])), ac=p[12])
    for k, v in R.items(): assert sum(int(v["ac"][i]) for i in CRIT) == v["criticos"], k
    return R

def report(name, R, seeds, rivals):
    print(f"\n=== {name} ===")
    for m in ["azar"] + rivals:
        v = [R[(m, s)]["criticos"] for s in seeds]
        print(f"{m:7s} critical per seed {v}  mean {np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}")
    for m in [r for r in ["azar", "olvido"] if (r, seeds[0]) in R]:
        b = sum(R[("juez", s)]["ac"][i] == "1" and R[(m, s)]["ac"][i] == "0" for s in seeds for i in CRIT)
        c = sum(R[("juez", s)]["ac"][i] == "0" and R[(m, s)]["ac"][i] == "1" for s in seeds for i in CRIT)
        print(f"McNemar judge vs {m}: judge-only {b}, {m}-only {c}, exact p = {binomtest(b, b + c, 0.5).pvalue:.2e}")

Q = read(["../logs/e21b_qwen_seed0.log", "../logs/e21b_qwen_seeds1-4.log"])
report("Qwen2.5-0.5B-Instruct (E21b, seeds 0-4)", Q, [0, 1, 2, 3, 4], ["olvido", "juez"])
L = read(["../logs/e21c_llama_seeds0-2.log"], after="E21c (Llama")
report("Llama-3.2-1B-Instruct (E21c, seeds 0-2)", L, [0, 1, 2], ["juez"])
