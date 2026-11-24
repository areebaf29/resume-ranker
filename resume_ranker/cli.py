"""Command line: rank a folder of resumes against a job description."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from .parsing import load_folder, read_text
from .ranker import ResumeRanker


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resume-ranker", description=__doc__)
    ap.add_argument("--resumes", required=True, help="Folder of .pdf/.docx/.txt/.md resumes")
    ap.add_argument("--job", required=True, help="Job description text file (.txt/.md/.pdf/.docx)")
    ap.add_argument("--top", type=int, default=10, help="How many to show (default 10)")
    ap.add_argument("--csv", help="Also write the ranking to this CSV file")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    names, texts = load_folder(args.resumes)
    if not texts:
        print("No readable resumes found.", file=sys.stderr)
        return 1
    t1 = time.perf_counter()
    jd = read_text(Path(args.job))
    ranker = ResumeRanker().fit(texts, names)
    t2 = time.perf_counter()
    results = ranker.rank(jd, top_k=args.top)
    t3 = time.perf_counter()

    print(f"{'#':>2}  {'score':>5}  {'text':>5} {'skill':>5} {'exp':>4}  resume")
    for r in results:
        print(f"{r.rank:>2}  {r.score:5.3f}  {r.text_score:5.2f} {r.skill_score:5.2f} "
              f"{r.experience_score:4.1f}  {r.name}")
        print(f"      matched: {', '.join(r.matched_skills) or '-'}")
        print(f"      missing: {', '.join(r.missing_skills) or '-'}")
    print(f"\n{len(texts)} resumes | read {t1-t0:.2f}s | index {t2-t1:.2f}s | rank {1000*(t3-t2):.1f}ms")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "resume", "score", "text", "skills", "experience", "matched", "missing"])
            for r in results:
                w.writerow([r.rank, r.name, f"{r.score:.4f}", f"{r.text_score:.4f}", f"{r.skill_score:.4f}",
                            f"{r.experience_score:.2f}", "; ".join(r.matched_skills), "; ".join(r.missing_skills)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
