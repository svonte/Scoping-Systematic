

#!/usr/bin/env python3
"""Initialize / update screening decisions (JSONL).

This script creates (or updates) a single audit-friendly decisions file:
  runs/decisions.jsonl

It scans all PubMed runs under a runs/ folder (each run contains records.json produced by pubmed1.py),
collects unique PMIDs, and ensures there is at least one decision entry per PMID for a given stage.

By default it only *adds missing* decisions (does not overwrite existing ones).

Usage examples:
  python3 pubmed1decisions.py
  python3 pubmed1decisions.py runs --stage abstract --default-decision maybe
  python3 pubmed1decisions.py runs --out runs/decisions.jsonl --dry-run

Later, Qwen (or you) can append refined decisions (include/exclude/maybe + reason_code).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_records_json(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: missing or invalid 'records' list")
    return records


def _find_records_files(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {runs_dir}")
    files = sorted(runs_dir.rglob("records.json"))
    if not files:
        raise FileNotFoundError(
            f"No records.json found under {runs_dir}. Expected runs/<query_id>/records.json"
        )
    return files


def _iter_pmids_from_runs(runs_dir: Path) -> Iterable[str]:
    for fp in _find_records_files(runs_dir):
        recs = _load_records_json(fp)
        for r in recs:
            if not isinstance(r, dict):
                continue
            pmid = r.get("pmid")
            if pmid:
                yield str(pmid)


def _read_existing_decisions(decisions_path: Path) -> Set[Tuple[str, str]]:
    """Return a set of (pmid, stage) pairs already present in decisions.jsonl."""
    existing: Set[Tuple[str, str]] = set()
    if not decisions_path.exists():
        return existing

    with decisions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{decisions_path}:{line_no}: invalid JSON")
            if not isinstance(obj, dict):
                continue
            pmid = obj.get("pmid")
            stage = obj.get("stage")
            if pmid and stage:
                existing.add((str(pmid), str(stage)))
    return existing


def _make_decision(pmid: str, stage: str, decision: str, by: str) -> Dict[str, Any]:
    return {
        "pmid": pmid,
        "stage": stage,
        "decision": decision,
        "reason_code": None,
        "by": by,
        "confidence": None,
        "timestamp": _utc_now_iso(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Initialize/update runs/decisions.jsonl with one entry per PMID (missing only)."
    )
    ap.add_argument(
        "runs_dir",
        nargs="?",
        default="runs",
        help="Path to runs/ folder (default: ./runs)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Decisions JSONL path (default: <runs_dir>/decisions.jsonl)",
    )
    ap.add_argument(
        "--stage",
        default="abstract",
        choices=["abstract", "fulltext"],
        help="Stage for the initialized decisions (default: abstract)",
    )
    ap.add_argument(
        "--default-decision",
        default="maybe",
        choices=["include", "exclude", "maybe"],
        help="Decision to assign to missing PMIDs (default: maybe)",
    )
    ap.add_argument(
        "--by",
        default="init",
        help="Value for the 'by' field (default: init)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not write anything; just report what would be added",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (runs_dir / "decisions.jsonl").resolve()

    pmids = sorted(set(_iter_pmids_from_runs(runs_dir)))
    existing = _read_existing_decisions(out_path)

    to_add = [p for p in pmids if (p, args.stage) not in existing]

    print(f"Runs dir: {runs_dir}")
    print(f"Unique PMIDs found: {len(pmids)}")
    print(f"Decisions file: {out_path}")
    print(f"Existing decisions for stage='{args.stage}': {sum(1 for _p,_s in existing if _s==args.stage)}")
    print(f"Missing decisions to add: {len(to_add)}")

    if args.dry_run:
        if to_add:
            print("DRY-RUN sample (first 3):")
            for pmid in to_add[:3]:
                print(json.dumps(_make_decision(pmid, args.stage, args.default_decision, args.by), ensure_ascii=False))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for pmid in to_add:
            obj = _make_decision(pmid, args.stage, args.default_decision, args.by)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"WROTE: appended {len(to_add)} decision lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())