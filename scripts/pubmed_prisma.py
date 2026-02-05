#!/usr/bin/env python3
"""Minimal PRISMA counter (Step 1).

Reads one or more PubMed `records.json` files (produced by pubmed1.py) under a `runs/` folder,
prints total retrieved records (pre-dedup), and computes unique records after deduplication by PMID.
Optionally saves a JSON summary for later PRISMA reporting.

Usage examples:
  python3 pubmed1prisma.py runs/
  python3 pubmed1prisma.py /abs/path/to/runs/ --out runs/prisma_step1_counts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from typing import Any, List

# Conservative defaults for "automation tools" exclusions (PRISMA 2020).
# These are applied only at the *pre-screening* stage.
DEFAULT_INELIGIBLE_PUBTYPES = {
    "editorial",
    "letter",
    "comment",
    "news",
    "interview",
    "published erratum",
    "retracted publication",
    "retraction of publication",
}


def _load_records_json(path: Path) -> List[dict[str, Any]]:
    """Load a single records.json and return the list under the 'records' key."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at top-level")

    records = data.get("records")
    if records is None:
        raise ValueError(f"{path}: missing key 'records'")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 'records' must be a list")

    return records


def find_records_files(runs_dir: Path) -> List[Path]:
    """Find all records.json files under runs_dir (recursively)."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {runs_dir}")

    files = sorted(runs_dir.rglob("records.json"))
    if not files:
        raise FileNotFoundError(
            f"No records.json found under {runs_dir}. Expected structure like runs/<query_id>/records.json"
        )
    return files


def _dedup_pmids(records: List[dict[str, Any]]) -> tuple[int, int, List[str]]:
    """Return (unique_count, duplicate_count, missing_pmid_ids).

    Deduplicates by PMID. Records missing PMID are not deduplicated and are tracked.
    """
    seen: set[str] = set()
    duplicates = 0
    missing: List[str] = []

    for r in records:
        pmid = r.get("pmid") if isinstance(r, dict) else None
        if not pmid:
            # track something identifiable if present
            title = r.get("title") if isinstance(r, dict) else None
            missing.append(title or "<missing title>")
            continue
        if pmid in seen:
            duplicates += 1
        else:
            seen.add(pmid)

    unique = len(seen) + len(missing)  # keep missing-PMID records as unique for counting purposes
    return unique, duplicates, missing


# --- AUTOMATION EXCLUSION HELPERS ---

def _norm_str(x: Any) -> str:
    return str(x).strip().lower()


def _automation_ineligible_reason(
    r: dict[str, Any],
    *,
    exclude_nonhuman: bool,
    exclude_pubtypes: bool,
    exclude_no_abstract: bool,
    exclude_non_english: bool,
) -> str | None:
    """Return a reason_code if record should be removed by automation rules, else None.

    Reasons are intentionally coarse and PRISMA-friendly.
    """
    # 1) No abstract (optional). Some workflows prefer to keep these for full-text retrieval.
    if exclude_no_abstract:
        abs_txt = r.get("abstract")
        if not abs_txt or not str(abs_txt).strip():
            return "NO_ABSTRACT"

    # 2) Non-human (optional) based on MeSH Humans flag when present.
    if exclude_nonhuman:
        is_humans = r.get("is_humans")
        if is_humans is False:
            return "ANIMAL_OR_IN_VITRO"

    # 3) Publication types (optional). Uses PubMed PublicationTypeList.
    if exclude_pubtypes:
        pts = r.get("publication_types") or []
        pts_norm = {_norm_str(p) for p in pts}
        if pts_norm & DEFAULT_INELIGIBLE_PUBTYPES:
            return "WRONG_DESIGN"

    # 4) Language (optional): exclude non-English.
    if exclude_non_english:
        langs = r.get("languages") or []
        langs_norm = {_norm_str(l) for l in langs}
        # PubMed languages are usually ISO codes like 'eng'.
        if langs_norm and "eng" not in langs_norm and "en" not in langs_norm and "english" not in langs_norm:
            return "LANGUAGE"

    return None


def _read_decisions_jsonl(path: Path) -> List[dict[str, Any]]:
    """Read decisions.jsonl (JSON Lines). Returns list of dicts."""
    decisions: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{line_no}: invalid JSON")
            if isinstance(obj, dict):
                decisions.append(obj)
    return decisions


def _latest_decision_by_pmid_stage(decisions: List[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Return mapping (pmid, stage) -> latest decision object.

    Uses 'timestamp' when parseable, otherwise last occurrence wins.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    latest_ts: dict[tuple[str, str], datetime] = {}

    for obj in decisions:
        pmid = obj.get("pmid")
        stage = obj.get("stage")
        if not pmid or not stage:
            continue
        key = (str(pmid), str(stage))

        ts_raw = obj.get("timestamp")
        ts = None
        if isinstance(ts_raw, str):
            try:
                # Accept ISO strings; datetime.fromisoformat handles offsets like +00:00
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = None

        if ts is None:
            # fallback: last line wins
            latest[key] = obj
            continue

        prev = latest_ts.get(key)
        if prev is None or ts >= prev:
            latest_ts[key] = ts
            latest[key] = obj

    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description="PRISMA step-1: count total PubMed records across runs")
    ap.add_argument(
        "runs_dir",
        nargs="?",
        default="runs",
        help="Path to runs/ folder (default: ./runs)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Path to write a JSON summary (default: <runs_dir>/prisma_step1_counts.json)",
    )
    ap.add_argument(
        "--no-out",
        action="store_true",
        default=False,
        help="Do not write the JSON summary (override default output)",
    )
    ap.add_argument(
        "--exclude-nonhuman",
        action="store_true",
        default=True,
        help="Automation: remove records with is_humans=False (default: enabled)",
    )
    ap.add_argument(
        "--exclude-pubtypes",
        action="store_true",
        default=True,
        help="Automation: remove Editorial/Letter/Comment/etc. via publication_types (default: enabled)",
    )
    ap.add_argument(
        "--exclude-no-abstract",
        action="store_true",
        default=False,
        help="Automation: remove records with missing abstract (default: disabled)",
    )
    ap.add_argument(
        "--exclude-non-english",
        action="store_true",
        default=False,
        help="Automation: remove non-English records (default: disabled)",
    )
    ap.add_argument(
        "--decisions",
        default=None,
        help="Optional decisions JSONL path (default: <runs_dir>/decisions.jsonl if it exists)",
    )
    ap.add_argument(
        "--stage",
        default="abstract",
        choices=["abstract", "fulltext"],
        help="Decision stage to count from decisions.jsonl (default: abstract)",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    record_files = find_records_files(runs_dir)

    all_records: List[dict[str, Any]] = []
    total = 0
    per_run = []
    for fp in record_files:
        records = _load_records_json(fp)
        all_records.extend(records)
        n = len(records)
        total += n
        per_run.append((fp, n))

    print(f"Runs dir: {runs_dir}")
    print(f"Found records.json files: {len(record_files)}")
    for fp, n in per_run:
        # show relative path for readability
        rel = fp.relative_to(runs_dir)
        print(f"  - {rel}: {n} records")
    print(f"TOTAL (pre-dedup): {total} records")

    unique, dup_removed, missing_pmids = _dedup_pmids(all_records)
    print(f"UNIQUE (post-dedup by PMID): {unique} records")
    print(f"DUPLICATE records removed: {dup_removed}")
    if missing_pmids:
        print(f"WARNING: {len(missing_pmids)} records missing PMID (counted as unique)")

    # Automation-based pre-screen removals (PRISMA: "Records marked as ineligible by automation tools")
    auto_reason_counts: dict[str, int] = {}
    auto_removed_pmids: set[str] = set()
    for r in all_records:
        if not isinstance(r, dict):
            continue
        reason = _automation_ineligible_reason(
            r,
            exclude_nonhuman=args.exclude_nonhuman,
            exclude_pubtypes=args.exclude_pubtypes,
            exclude_no_abstract=args.exclude_no_abstract,
            exclude_non_english=args.exclude_non_english,
        )
        if reason:
            auto_reason_counts[reason] = auto_reason_counts.get(reason, 0) + 1
            pmid = r.get("pmid")
            if pmid:
                auto_removed_pmids.add(str(pmid))

    auto_removed_n = sum(auto_reason_counts.values())
    # For PRISMA counting, apply automation removals on the unique-by-PMID universe.
    # If a PMID appears multiple times across runs, it has already been deduplicated.
    auto_removed_unique = len(auto_removed_pmids)

    print(f"AUTOMATION ineligible (pre-screen): {auto_removed_unique} records")
    if auto_reason_counts:
        for k in sorted(auto_reason_counts):
            print(f"  - {k}: {auto_reason_counts[k]}")

    screened_n = unique - auto_removed_unique
    print(f"RECORDS TO SCREEN (title/abstract): {screened_n} records")

    # Optional: Screening counts from decisions.jsonl (does not affect Step-1 counts)
    decisions_path = Path(args.decisions).expanduser().resolve() if args.decisions else (runs_dir / "decisions.jsonl").resolve()
    screening_available = decisions_path.exists()

    screened_counts: dict[str, int] = {}
    excluded_by_reason: dict[str, int] = {}

    if screening_available:
        decisions = _read_decisions_jsonl(decisions_path)
        latest = _latest_decision_by_pmid_stage(decisions)

        # Build the universe of PMIDs that are eligible for screening (unique, post automation)
        screened_universe: set[str] = set()
        for r in all_records:
            if not isinstance(r, dict):
                continue
            pmid = r.get("pmid")
            if not pmid:
                continue
            pmid_s = str(pmid)
            if pmid_s in auto_removed_pmids:
                continue
            screened_universe.add(pmid_s)

        # Count latest decisions for those PMIDs at the requested stage
        for pmid in screened_universe:
            obj = latest.get((pmid, args.stage))
            if not obj:
                screened_counts["missing_decision"] = screened_counts.get("missing_decision", 0) + 1
                continue
            dec = str(obj.get("decision") or "missing_decision").lower()
            screened_counts[dec] = screened_counts.get(dec, 0) + 1

            if dec == "exclude":
                rc = obj.get("reason_code")
                rc_s = str(rc) if rc else "UNSPECIFIED"
                excluded_by_reason[rc_s] = excluded_by_reason.get(rc_s, 0) + 1

        print(f"SCREENING decisions: {decisions_path}")
        print(f"SCREENED (universe): {len(screened_universe)} records")
        for k in sorted(screened_counts):
            print(f"  - {k}: {screened_counts[k]}")
        if excluded_by_reason:
            print("EXCLUDED by reason_code:")
            for k in sorted(excluded_by_reason):
                print(f"  - {k}: {excluded_by_reason[k]}")
    else:
        print(f"SCREENING decisions: not found ({decisions_path}); skipping screening counts")

    # Write JSON summary by default unless explicitly disabled.
    if not args.no_out:
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
        else:
            out_path = (runs_dir / "prisma_step1_counts.json").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "runs_dir": str(runs_dir),
            "n_runs": len(record_files),
            "per_run": [
                {"records_json": str(fp.relative_to(runs_dir)), "n_records": n}
                for fp, n in per_run
            ],
            "total_pre_dedup": total,
            "unique_post_dedup_pmid": unique,
            "duplicates_removed": dup_removed,
            "missing_pmid_count": len(missing_pmids),
            "automation_ineligible_unique": auto_removed_unique,
            "automation_ineligible_by_reason": auto_reason_counts,
            "records_to_screen": screened_n,
            "screening": {
                "available": screening_available,
                "decisions_path": str(decisions_path),
                "stage": args.stage,
                "counts": screened_counts,
                "excluded_by_reason": excluded_by_reason,
            },
        }
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"WROTE: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())