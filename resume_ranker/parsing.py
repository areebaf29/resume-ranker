"""Read resumes from .txt, .md, .pdf and .docx files."""
from __future__ import annotations

import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def read_text(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx

        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.extend(c.text for c in row.cells)
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_folder(folder: str | Path) -> tuple[list[str], list[str]]:
    """Return (names, texts) for every supported file in a folder (recursive).

    Unreadable files are skipped with a warning on stderr, so one bad PDF
    does not stop a run of thousands.
    """
    import sys

    names, texts = [], []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            try:
                t = read_text(p)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"warning: skipped {p.name}: {exc}", file=sys.stderr)
                continue
            if t.strip():
                names.append(str(p.relative_to(folder)))
                texts.append(t)
            else:
                print(f"warning: skipped {p.name}: no extractable text", file=sys.stderr)
    return names, texts


_YEARS = re.compile(r"(\d{1,2})(?:\.\d)?\s*\+?\s*(?:years?|yrs?)\b", re.I)
_EXP_CONTEXT = re.compile(r"experience|worked|working|internship|industry|professional|employed|exp\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d")
_URL = re.compile(r"https?://\S+|www\.\S+|(?:linkedin|github)\.com/\S+", re.I)


def max_years(text: str) -> float:
    """Largest 'N years' figure that sits in a sentence about experience.

    "3 years of experience" counts. "founded 20 years ago" or "10 years old" does
    not, because the sentence has no experience wording. Still a rough signal.
    """
    vals = []
    for sent in re.split(r"[.\n;]", text):
        if _EXP_CONTEXT.search(sent):
            vals.extend(int(m.group(1)) for m in _YEARS.finditer(sent))
    vals = [v for v in vals if v <= 40]
    return float(max(vals)) if vals else 0.0


def redact_pii(text: str) -> str:
    """Remove emails, phone numbers and profile URLs. Names are not detected."""
    return _PHONE.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", text)))
