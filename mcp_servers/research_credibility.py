"""Credibility is code, not opinion: one deterministic rule, written down here and nowhere else.

The same function scores a search hit (OpenAlex metadata only) and a full record from get_work
(OpenAlex + Crossref notices), so every harness that uses the research server sees the same numbers.

    score (0-100) = kind     45  peer-reviewed journal article or review (a journal venue AND type article/review)
                             20  other edited scholarly work (book chapter, conference paper, book)
                              0  preprint, thesis, dataset, erratum, editorial, letter, anything else
                  + venue    15  the venue is known and has an ISSN (a real journal or series, not a repository)
                  + cited  0-25  25 * log10(1 + citations) / 3, capped: 1,000+ citations = 25, 100 = 17, 10 = 9
                  + recent 0-15  15 if published in the last 10 years, 8 if 10-20 years ago, 0 if older
                  - concern  30  an expression of concern or a partial retraction has been published
    A retracted or withdrawn work scores 0, whatever else is true.
    tier: high >= 70, medium >= 50, else low.
    vetted (allowed in a report) = peer-reviewed AND not retracted AND score >= 60.

"Peer-reviewed" is a proxy: no open API records peer review itself, so we use "journal-type venue with an
ISSN + research article or review" and never let a model decide it. Corrections are normal science:
they are reported, not penalised.
"""
import math
import time

PEER_REVIEWED_TYPES = {"article", "review"}
EDITED_TYPES = {"book-chapter", "book", "proceedings-article", "proceedings", "monograph", "reference-entry"}
NOT_RESEARCH = {"erratum", "editorial", "letter", "paratext", "retraction", "peer-review", "supplementary-materials"}
VETTED_MIN = 60


def credibility(work: dict, as_of: int | None = None) -> dict:
    as_of = as_of or time.gmtime().tm_year
    typ, venue_type = (work.get("type") or "").lower(), (work.get("venue_type") or "").lower()
    crossref_type = (work.get("crossref_type") or "").lower()
    updates = {u.get("type") for u in work.get("updates") or []}
    retracted = bool(work.get("is_retracted")) or bool(updates & {"retraction", "withdrawal", "removal"})
    peer_reviewed = (typ in PEER_REVIEWED_TYPES and venue_type == "journal" and bool(work.get("venue"))
                     and crossref_type in ("", "journal-article") and not work.get("is_update_notice"))
    reasons = []
    if peer_reviewed:
        kind = 45
        reasons.append(f"peer-reviewed journal {typ} (+45)")
    elif typ in EDITED_TYPES or crossref_type in EDITED_TYPES:
        kind = 20
        reasons.append(f"edited scholarly work: {typ or crossref_type} (+20)")
    else:
        kind = 0
        reasons.append(f"not a peer-reviewed research article: {typ or crossref_type or 'unknown type'} (+0)")
    venue = 15 if (work.get("venue") and work.get("issn") and venue_type in ("journal", "conference", "book series")) else 0
    reasons.append(f"venue {work.get('venue') or 'unknown'}" + (" with ISSN (+15)" if venue else " not a known journal (+0)"))
    cites = int(work.get("cited_by_count") or 0)
    cited = min(25.0, 25 * math.log10(1 + cites) / 3)
    reasons.append(f"{cites} citations (+{cited:.1f})")
    year = work.get("year")
    age = as_of - year if year else None
    recent = 0 if age is None else 15 if age <= 10 else 8 if age <= 20 else 0
    reasons.append(f"published {year or 'year unknown'} (+{recent})")
    concern = 30 if updates & {"expression_of_concern", "partial_retraction"} else 0
    if concern:
        reasons.append("expression of concern or partial retraction (-30)")
    if updates & {"correction", "corrigendum", "erratum"}:
        reasons.append("a correction notice exists (no penalty; read it)")
    score = 0.0 if retracted else max(0.0, kind + venue + cited + recent - concern)
    if retracted:
        reasons.insert(0, "RETRACTED: score 0, never cite")
    if typ in NOT_RESEARCH:
        reasons.append(f"{typ} is a notice, not a study")
    score = round(score, 1)
    return {"score": score, "tier": "high" if score >= 70 else "medium" if score >= 50 else "low",
            "peer_reviewed": peer_reviewed, "retracted": retracted,
            "vetted": peer_reviewed and not retracted and score >= VETTED_MIN, "reasons": reasons}
