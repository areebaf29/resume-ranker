"""Skill taxonomy and single-pass skill extraction."""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "data" / "skills.json"


class SkillExtractor:
    """Finds canonical skills in text using one compiled regex."""

    def __init__(self, taxonomy: dict[str, list[str]] | None = None):
        if taxonomy is None:
            taxonomy = json.loads(DEFAULT_TAXONOMY.read_text(encoding="utf-8"))
        self.skills: list[str] = sorted(taxonomy)
        self.index = {s: i for i, s in enumerate(self.skills)}
        alias_to_skill: dict[str, str] = {}
        for skill, aliases in taxonomy.items():
            alias_to_skill[skill.lower()] = skill
            for a in aliases:
                alias_to_skill[a.lower()] = skill
        self._alias = alias_to_skill
        aliases_sorted = sorted(alias_to_skill, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in aliases_sorted)
        # Boundaries that treat + # . as part of a word, so "c" never matches inside "c++".
        self._rx = re.compile(rf"(?<![A-Za-z0-9+#])(?:{body})(?![A-Za-z0-9+#])", re.I)

    # Words right before a skill that mean the candidate does NOT have it.
    _NEG = re.compile(
        r"(?:\b(?:no|not|without|lack(?:s|ing)?(?: of)?|never|zero)\b"
        r"(?:\s+(?:prior|previous|any|formal|hands-on|work|professional|direct|much))*"
        r"(?:\s+(?:experience|exposure|knowledge|background|familiarity)(?:\s+(?:with|in|of))?)?"
        r"|\bunfamiliar with|\bhave not used|\bhaven't used|\bdon't know)\s*(?:\w+\s+){0,1}$",
        re.I)

    def extract(self, text: str, skip_negated: bool = False) -> set[str]:
        """Canonical skills found in text.

        With skip_negated, a skill is ignored when the few words right before it
        say the candidate lacks it ("no experience with Tableau"). This is a
        small rule list, not language understanding.
        """
        found = set()
        for m in self._rx.finditer(text):
            if skip_negated:
                start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start()), 0)
                if self._NEG.search(text[start:m.start()]):
                    continue
            found.add(self._alias[m.group(0).lower()])
        return found

    def extract_ids(self, text: str, skip_negated: bool = False) -> list[int]:
        return sorted(self.index[s] for s in self.extract(text, skip_negated))
