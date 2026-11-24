"""Synthetic resumes and job descriptions with known ground truth.

There is no public labelled resume/job dataset bundled here, so quality is
measured on generated data. Each candidate has a hidden set of true skills.
Resume text mentions those skills using random aliases, inside varied
sentences, mixed with filler and skills outside the taxonomy. A job's
relevance for a candidate is the share of the job's required skills the
candidate truly has. These numbers show the ranker behaves sensibly. They do
not predict accuracy on real resumes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TAX = json.loads((Path(__file__).resolve().parent.parent / "data" / "skills.json").read_text())

ROLES = {
    "data analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "statistics", "data visualization", "python", "pandas"],
    "ml engineer": ["python", "machine learning", "scikit-learn", "pytorch", "tensorflow", "docker", "mlops", "pandas", "numpy", "git"],
    "data engineer": ["python", "sql", "spark", "airflow", "etl", "aws", "docker", "hadoop", "postgresql", "git"],
    "backend developer": ["java", "rest api", "postgresql", "docker", "git", "linux", "django", "mysql", "kubernetes"],
    "frontend developer": ["javascript", "typescript", "react", "html", "css", "git", "node.js", "rest api"],
    "nlp engineer": ["python", "nlp", "llm", "rag", "pytorch", "generative ai", "machine learning", "fastapi", "docker"],
    "devops engineer": ["docker", "kubernetes", "aws", "linux", "ci/cd", "git", "azure", "python", "gcp"],
}
NOISE = ["community volunteering", "public speaking", "photography", "chess club", "team leadership",
         "stakeholder reports", "documentation", "mentoring juniors", "hackathon participation",
         "inventory systems", "customer support", "campus ambassador"]
OUTSIDE = ["figma", "tally", "wordpress", "salesforce", "sap", "autocad", "jira", "slack"]
TEMPLATES = [
    "Built {s} based solutions for a team project and presented the results.",
    "Hands-on experience with {s} during a {m}-month internship.",
    "Used {s} to deliver weekly work for stakeholders.",
    "Skills: {s}.",
    "Completed coursework and a capstone project using {s}.",
]


def make_resume(rng: random.Random, role: str, skill_pool_extra: int = 3):
    core = ROLES[role]
    k = rng.randint(max(2, len(core) // 3), len(core))
    own = set(rng.sample(core, k))
    others = [s for s in TAX if s not in core]
    own |= set(rng.sample(others, skill_pool_extra))
    lines = [f"Candidate {rng.randint(0, 10**6)}", f"Aspiring {role}" if rng.random() < .5 else f"{role.title()}"]
    for s in own:
        alias = rng.choice([s] + TAX[s])
        lines.append(rng.choice(TEMPLATES).format(s=alias, m=rng.randint(2, 12)))
    lines += [f"Interests: {', '.join(rng.sample(NOISE, 3))}.", f"Also familiar with {rng.choice(OUTSIDE)}."]
    if rng.random() < .5:
        lines.append(f"{rng.randint(0, 6)} years of experience.")
    rng.shuffle(lines)
    return "\n".join(lines), own


def make_job(rng: random.Random, role: str):
    core = ROLES[role]
    req = rng.sample(core, rng.randint(4, min(7, len(core))))
    pref = [s for s in rng.sample(core, 2) if s not in req][:1]
    jd = [f"We are hiring a {role}.", "Requirements:"]
    jd += [f"- Experience with {rng.choice([s] + TAX[s])}" for s in req]
    jd += ["Nice to have:"] + [f"- {p}" for p in pref]
    jd.append("You will work with stakeholders and deliver reliable work.")
    return "\n".join(jd), set(req)


def build(n_resumes=2000, n_jobs=40, seed=7):
    rng = random.Random(seed)
    roles = list(ROLES)
    texts, owned = [], []
    for i in range(n_resumes):
        t, o = make_resume(rng, roles[i % len(roles)])
        texts.append(t)
        owned.append(o)
    jobs = []
    for j in range(n_jobs):
        jd, req = make_job(rng, roles[j % len(roles)])
        rel = [len(req & o) / len(req) for o in owned]
        jobs.append((jd, rel))
    return texts, jobs
