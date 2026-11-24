# Resume Ranker

Ranks a folder of resumes (PDF, DOCX, TXT, MD) against a job description and
explains each score: which required skills matched and which are missing.

Built for speed: resumes are indexed once into sparse matrices, so ranking a
job description is one sparse matrix product plus a top-k selection.

## How it scores

Each resume gets a score from 0 to 1 that combines three signals:

| Signal | Weight | What it measures |
| --- | --- | --- |
| Text match | 0.35 | BM25 between the job description and the resume text |
| Skill coverage | 0.55 | Share of the job's skills found in the resume. Skills under "nice to have" or "preferred" count half. |
| Experience | 0.10 | Largest "N years" figure in the resume against the smallest "N years" in the job |

If the job description has no recognised skills or no years requirement, that
signal is dropped and the others are rescaled. Skills come from
`data/skills.json` (about 60 skills with aliases such as `sklearn`, `k8s`,
`C#`). Add your own entries to extend it.

## Install

```bash
pip install -r requirements.txt
```

## Use

```bash
python -m resume_ranker --resumes examples/resumes --job examples/job_description.txt --top 5 --csv ranking.csv
```

From Python:

```python
from resume_ranker import ResumeRanker

ranker = ResumeRanker().fit(texts, names)      # index once
for r in ranker.rank(job_description, top_k=10):
    print(r.rank, r.name, round(r.score, 3), r.matched_skills, r.missing_skills)
```

Unreadable or empty files are skipped with a warning. Scanned PDFs without a
text layer are not supported (no OCR).

## Measured results

Run `python -m benchmarks.evaluate`. Numbers from one run (2,000 synthetic
resumes, 40 jobs, seed 7):

| Method | nDCG@10 | P@10 |
| --- | --- | --- |
| TF-IDF cosine (baseline) | 0.811 | 0.835 |
| BM25 only | 0.770 | 0.787 |
| Skills only | 0.992 | 0.993 |
| Hybrid (default weights) | 0.991 | 0.990 |

Speed on a single CPU core (indexing includes skill extraction):

| Resumes | Index time | Rank one job |
| --- | --- | --- |
| 1,000 | 0.3 s | about 1 ms |
| 10,000 | 2.9 s | about 2 ms |
| 50,000 | 14.9 s | about 5 ms |

### Read these numbers carefully

- The data is **synthetic**. Resumes are generated from hidden skill sets, and
  relevance is the share of required skills a candidate truly has. The skill
  signal uses the same taxonomy that generated the text, so its score is
  optimistic. On real resumes, expect lower numbers.
- The only fair conclusion is the gap over plain TF-IDF/BM25 on this setup, and
  that ranking is fast. Accuracy on real hiring data has not been measured, and
  there is no labelled real dataset in this repo.
- Default weights were set by hand, not tuned on the benchmark.
- It matches words and listed skills. It does not understand context (for
  example "no experience with SQL" still counts as SQL) and it can be gamed by
  keyword stuffing. Use it to shortlist for human review, not to reject people.

## Fairness and privacy

- The score uses only skills, years of experience and text overlap with the job.
  It does not read or score names, gender, age, photos, college names or
  location as features. But the text match step still sees all words, so
  proxies can leak in (for example a college name that appears in the job
  description). Review shortlists for that, and do not use the score as the
  only basis for rejecting anyone.
- Emails, phone numbers and profile URLs are removed before indexing
  (`redact=True`, default). Names and addresses are NOT detected.
- Everything runs locally. Nothing is uploaded, and the index lives in memory only.
- Output CSVs are ignored by git (`*.csv`) so rankings of real people are not committed by accident.
- Not audited for bias. Skill aliases are English only.

## Behaviour notes

- Ranking is deterministic: ties break by input order, and `top_k=5` is always the
  first five of `top_k=50`. Scores depend on the indexed batch (BM25 weights rare
  terms more), so the same resume can get a different score when indexed with
  a different set of resumes. Order within one batch is stable.
- A skill is ignored when it is negated just before it ("no experience with
  Tableau"). This is a small rule list, so phrasing like "Tableau: none" is
  still counted.
- Years of experience count only in sentences about experience, so "founded 20
  years ago" is ignored. Date ranges ("2021 to 2024") are not converted to years.

## Tests

13 tests cover: skill extraction with symbols (C++, C#, node.js), ranking order
and explanations, empty/unmatched job text, empty resume list, bad inputs,
unreadable/empty/unsupported files, no-resumes CLI exit, negated skills,
unrelated numbers vs years, PII redaction, and top-k/rebuild stability.

```bash
python -m pytest -q
```

## Project layout

```
resume_ranker/   parsing, skill extraction, ranker, CLI
data/skills.json skill taxonomy and aliases
benchmarks/      synthetic data generator and evaluation
tests/           unit tests
examples/        a sample job description and three synthetic resumes
```
