"""Ranking quality and speed benchmark on synthetic data (see synth.py).

Usage: python -m benchmarks.evaluate [--resumes 2000] [--jobs 40] [--seed 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from benchmarks.synth import build  # noqa: E402
from resume_ranker import ResumeRanker  # noqa: E402


def ndcg_at_k(scores, rel, k=10):
    rel = np.asarray(rel)
    gains = np.round(3 * rel)  # graded 0..3
    top = np.argsort(-scores, kind="stable")[:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = float(((2 ** gains[top] - 1) * disc).sum())
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(((2 ** ideal - 1) * disc).sum())
    return dcg / idcg if idcg else 0.0


def precision_at_k(scores, rel, k=10, thr=0.8):
    top = np.argsort(-scores, kind="stable")[:k]
    return float((np.asarray(rel)[top] >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    texts, jobs = build(a.resumes, a.jobs, a.seed)

    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    methods = {
        "TF-IDF cosine (baseline)": lambda jd: np.asarray((X @ tfidf.transform([jd]).T).todense()).ravel(),
    }
    full = ResumeRanker().fit(texts)
    methods["BM25 only"] = lambda jd: full.score_all(jd)[1]["text"]
    methods["Skills only"] = lambda jd: full.score_all(jd)[1]["skills"]
    methods["Hybrid (default weights)"] = lambda jd: full.score_all(jd)[0]

    print(f"{a.resumes} synthetic resumes, {a.jobs} jobs, seed {a.seed}")
    print(f"{'method':28} {'nDCG@10':>8} {'P@10':>6}")
    for name, fn in methods.items():
        nd = np.mean([ndcg_at_k(fn(jd), rel) for jd, rel in jobs])
        pr = np.mean([precision_at_k(fn(jd), rel) for jd, rel in jobs])
        print(f"{name:28} {nd:8.3f} {pr:6.3f}")

    print("\nSpeed (this machine, single thread)")
    for n in (1000, 10000, 50000):
        t, js = build(n, 5, 1)
        t0 = time.perf_counter()
        r = ResumeRanker().fit(t)
        t1 = time.perf_counter()
        lat = []
        for jd, _ in js:
            s = time.perf_counter()
            r.rank(jd, top_k=10)
            lat.append(time.perf_counter() - s)
        print(f"{n:>6} resumes: index {t1-t0:6.2f}s ({n/(t1-t0):7.0f}/s), "
              f"rank one job median {1000*np.median(lat):6.1f} ms")


if __name__ == "__main__":
    main()
