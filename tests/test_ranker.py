import numpy as np
import pytest

from resume_ranker import ResumeRanker
from resume_ranker.parsing import max_years
from resume_ranker.skills import SkillExtractor

JD = """Data Analyst
Requirements:
- 2+ years of experience
- SQL and Excel
- Power BI dashboards
Nice to have:
- Python
"""
STRONG = "Data analyst with 3 years experience. Built Power BI dashboards, wrote SQL queries, advanced Excel. Python scripts."
WEAK = "Frontend developer. React, HTML, CSS, TypeScript. 1 year experience."
MID = "Analyst using Excel and some SQL. 2 years of experience."


def test_skill_extraction_handles_symbols():
    ex = SkillExtractor()
    assert ex.extract("I write C++ and C# and node.js") >= {"c++", "c#", "node.js"}
    assert "r" not in ex.extract("c++ and python")
    assert "java" not in ex.extract("javascript only")
    assert "sql" in ex.extract("MySQL and SQL")


def test_best_resume_first_and_explained():
    r = ResumeRanker().fit([WEAK, MID, STRONG], ["weak", "mid", "strong"])
    out = r.rank(JD, top_k=3)
    assert [x.name for x in out] == ["strong", "mid", "weak"]
    assert "sql" in out[0].matched_skills and "power bi" in out[0].matched_skills
    assert "power bi" in out[1].missing_skills
    assert out[0].score > out[1].score > out[2].score


def test_scores_bounded_and_top_k_clamped():
    r = ResumeRanker().fit([WEAK, MID, STRONG])
    scores, _ = r.score_all(JD)
    assert np.all(scores >= 0) and np.all(scores <= 1.0001)
    assert len(r.rank(JD, top_k=50)) == 3


def test_empty_or_unmatched_job_does_not_crash():
    r = ResumeRanker().fit([WEAK, MID])
    assert len(r.rank("", top_k=2)) == 2
    assert len(r.rank("zzzz qqqq", top_k=2)) == 2


def test_deterministic_ties():
    r = ResumeRanker().fit(["python sql"] * 4)
    assert [x.name for x in r.rank("python sql", 4)] == ["resume_0", "resume_1", "resume_2", "resume_3"]


def test_input_validation():
    with pytest.raises(ValueError):
        ResumeRanker().fit([])
    with pytest.raises(ValueError):
        ResumeRanker().fit(["a"], ["x", "y"])


def test_years():
    assert max_years("5 years of experience and 2 yrs internship") == 5
    assert max_years("no numbers") == 0


def test_negated_skill_not_credited():
    ex = SkillExtractor()
    assert "tableau" not in ex.extract("No experience with Tableau.", skip_negated=True)
    assert "tableau" not in ex.extract("I am unfamiliar with Tableau", skip_negated=True)
    assert "tableau" in ex.extract("Experience with Tableau, no gaps.", skip_negated=True)
    r = ResumeRanker().fit(["No experience with SQL. Uses Excel.", "Uses SQL and Excel."], ["neg", "pos"])
    assert r.rank("Requirements: SQL, Excel", 2)[0].name == "pos"


def test_unrelated_numbers_not_years():
    assert max_years("Company founded 20 years ago. I am 22 years old.") == 0
    assert max_years("Graduated in 2024. 3 years of experience in analytics.") == 3


def test_pii_redacted_before_indexing():
    from resume_ranker.parsing import redact_pii
    t = redact_pii("Jane jane.doe@mail.com +91 98765 43210 https://linkedin.com/in/jane python")
    assert "@" not in t and "98765" not in t and "linkedin" not in t and "python" in t


def test_stable_across_topk_and_rebuild():
    texts = [f"python sql {i % 3} excel" if i % 2 else "java docker" for i in range(30)]
    jd = "Requirements: python, sql, excel"
    a = ResumeRanker().fit(texts)
    b = ResumeRanker().fit(texts)
    full = [x.name for x in a.rank(jd, 30)]
    assert full == [x.name for x in b.rank(jd, 30)]
    assert full[:5] == [x.name for x in a.rank(jd, 5)]


def test_unreadable_and_empty_files_skipped(tmp_path, capsys):
    from resume_ranker.parsing import load_folder
    (tmp_path / "good.txt").write_text("python sql")
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "broken.docx").write_bytes(b"not a docx")
    (tmp_path / "notes.xyz").write_text("ignored")
    names, texts = load_folder(tmp_path)
    assert names == ["good.txt"]
    assert "skipped" in capsys.readouterr().err


def test_cli_no_resumes(tmp_path):
    from resume_ranker.cli import main
    (tmp_path / "jd.txt").write_text("python")
    (tmp_path / "r").mkdir()
    assert main(["--resumes", str(tmp_path / "r"), "--job", str(tmp_path / "jd.txt")]) == 1
