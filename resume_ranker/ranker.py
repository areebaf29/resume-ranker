"""Hybrid resume ranker: BM25 text match + weighted skill coverage + experience.

Everything is precomputed into sparse matrices when resumes are indexed, so
each job description costs one sparse matrix-vector product plus a top-k
selection. No per-resume Python loop runs at query time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .parsing import max_years, redact_pii
from .skills import SkillExtractor

_PREFERRED = re.compile(r"preferred|nice to have|bonus|good to have|plus\b|optional", re.I)
_REQ_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)

DEFAULT_WEIGHTS = {"text": 0.35, "skills": 0.55, "experience": 0.10}


@dataclass
class RankedResume:
    rank: int
    name: str
    score: float
    text_score: float
    skill_score: float
    experience_score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


class ResumeRanker:
    def __init__(self, k1: float = 1.5, b: float = 0.75, weights: dict | None = None,
                 extractor: SkillExtractor | None = None, redact: bool = True):
        self.redact = redact
        self.k1, self.b = k1, b
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.extractor = extractor or SkillExtractor()
        self._vec = CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 1),
                                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]*[a-zA-Z0-9+#]\b|\b[a-zA-Z]\b",
                                    dtype=np.float32)
        self.names: list[str] = []

    # ---------------------------------------------------------------- index
    def fit(self, texts: list[str], names: list[str] | None = None) -> "ResumeRanker":
        if not texts:
            raise ValueError("No resumes to index.")
        if self.redact:  # contact details carry no job signal and are personal data
            texts = [redact_pii(t) for t in texts]
        self.names = list(names) if names is not None else [f"resume_{i}" for i in range(len(texts))]
        if len(self.names) != len(texts):
            raise ValueError("names and texts must have the same length")
        tf = self._vec.fit_transform(texts).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        self._idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = max(float(dl.mean()), 1.0)
        # BM25 term weight per stored (doc, term): idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl)
        data = self._idf[tf.indices] * tf.data * (self.k1 + 1) / denom
        self._bm25 = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
        self._bm25_csc = self._bm25.tocsc()  # column access: query touches few terms
        # skill presence matrix (docs x skills)
        ri, ci = [], []
        for i, t in enumerate(texts):
            ids = self.extractor.extract_ids(t, skip_negated=True)
            ri.extend([i] * len(ids))
            ci.extend(ids)
        self._skills = sp.csr_matrix((np.ones(len(ri), dtype=np.float32), (ri, ci)),
                                     shape=(n, len(self.extractor.skills)))
        self._years = np.array([max_years(t) for t in texts], dtype=np.float32)
        return self

    # ---------------------------------------------------------------- query
    def _job_skills(self, jd: str) -> tuple[np.ndarray, np.ndarray]:
        """Weights (1.0 required, 0.5 preferred) over the skill vocabulary."""
        w = np.zeros(len(self.extractor.skills), dtype=np.float32)
        for line in re.split(r"[\n.;]", jd):
            weight = 0.5 if _PREFERRED.search(line) else 1.0
            for s in self.extractor.extract(line):
                w[self.extractor.index[s]] = max(w[self.extractor.index[s]], weight)
        return w, np.flatnonzero(w)

    def rank(self, job_description: str, top_k: int = 10) -> list[RankedResume]:
        scores, parts = self.score_all(job_description)
        n = len(scores)
        k = min(top_k, n)
        # Keep every document tied with the k-th score, then order by (score desc, index asc),
        # so the result is identical for any top_k and does not depend on partition order.
        kth = np.partition(scores, n - k)[n - k]
        cand = np.flatnonzero(scores >= kth)
        top = cand[np.lexsort((cand, -scores[cand]))][:k]
        sw, sidx = self._job_skills(job_description)
        out = []
        for r, i in enumerate(top, 1):
            have = self._skills[i].indices
            have_set = set(have.tolist())
            matched = [self.extractor.skills[j] for j in sidx if j in have_set]
            missing = [self.extractor.skills[j] for j in sidx if j not in have_set]
            out.append(RankedResume(r, self.names[i], float(scores[i]), float(parts["text"][i]),
                                    float(parts["skills"][i]), float(parts["experience"][i]),
                                    matched, missing))
        return out

    def score_all(self, job_description: str) -> tuple[np.ndarray, dict]:
        n = self._bm25.shape[0]
        # text: BM25 with the JD as query (term frequency in query ignored)
        q = self._vec.transform([job_description])
        cols = q.indices
        if len(cols):
            text = np.asarray(self._bm25_csc[:, cols].sum(axis=1)).ravel()
            # Normalize by the best possible score a document could get for these terms,
            # so the value is comparable across job descriptions (0..~1).
            ceiling = float(self._idf[cols].sum() * (self.k1 + 1))
            text = np.clip(text / ceiling, 0, 1)
        else:
            text = np.zeros(n, dtype=np.float32)
        sw, sidx = self._job_skills(job_description)
        if len(sidx):
            skill = (self._skills[:, sidx] @ sw[sidx]) / sw[sidx].sum()
            skill = np.asarray(skill).ravel()
        else:
            skill = np.zeros(n, dtype=np.float32)
        ym = _REQ_YEARS.findall(job_description)
        if ym:
            need = max(1.0, float(min(int(y) for y in ym)))
            exp = np.clip(self._years / need, 0, 1)
        else:
            exp = np.zeros(n, dtype=np.float32)
        # Drop weights for signals the job description does not provide, then renormalize.
        w = dict(self.weights)
        if not len(sidx):
            w["skills"] = 0.0
        if not ym:
            w["experience"] = 0.0
        total = sum(w.values()) or 1.0
        score = (w["text"] * text + w["skills"] * skill + w["experience"] * exp) / total
        return score.astype(np.float32), {"text": text, "skills": skill, "experience": exp}
