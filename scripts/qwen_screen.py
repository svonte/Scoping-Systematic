#!/usr/bin/env python3
"""LLM-assisted title/abstract screening for remaining 'maybe' records.

This script:
- Reads all runs/<query_id>/records.json (produced by pubmed1.py)
- Builds the post-automation screening universe (unique PMIDs)
- Reads runs/decisions.jsonl (append-only audit log)
- Selects PMIDs whose latest decision at a stage is missing or 'maybe'
- Calls Ollama locally (ollama run <model>) to classify each record
- Appends new decision lines to decisions.jsonl with by='qwen'

Design goals: simple, local, reproducible, low-hallucination.

Usage:
  python3 pubmed1qwen_screen.py
  python3 pubmed1qwen_screen.py runs --max 20 --dry-run
  python3 pubmed1qwen_screen.py runs --model qwen3-coder-30b-screen
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import requests
import ast
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# Keep these aligned with pubmed1prisma.py automation defaults
DEFAULT_AUTOMATION_INELIGIBLE_PUBTYPES = {
    "editorial",
    "letter",
    "comment",
    "news",
    "interview",
    "published erratum",
    "retracted publication",
    "retraction of publication",
}

# Screening reason codes: keep small and PRISMA-friendly.
EXCLUSION_REASON_CODES = {
    "NOT_RELEVANT",
    "WRONG_DESIGN",
    "WRONG_POPULATION",
    "WRONG_OUTCOME",
    "WRONG_INTERVENTION",
    "NO_ABSTRACT",
    "LANGUAGE",
    "OTHER",
}
INCLUSION_REASON_CODES = {"ELIGIBLE", "POTENTIALLY_ELIGIBLE"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_str(x: Any) -> str:
    return str(x).strip().lower()

def _read_criteria_text(criteria_path: Path) -> str:
    """Read criteria text from a file and return it as plain text.

    This is the authoritative criteria block inserted into the LLM prompt.
    """
    if not criteria_path.exists():
        raise FileNotFoundError(f"Criteria file not found: {criteria_path}")
    txt = criteria_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Criteria file is empty: {criteria_path}")
    return txt + "\n"

def _load_records_json(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: missing or invalid 'records' list")
    return [r for r in records if isinstance(r, dict)]


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


def _iter_records(runs_dir: Path) -> Iterable[dict[str, Any]]:
    for fp in _find_records_files(runs_dir):
        for r in _load_records_json(fp):
            yield r


def _automation_ineligible(r: dict[str, Any], *, exclude_nonhuman: bool = True, exclude_pubtypes: bool = True) -> bool:
    if exclude_nonhuman:
        is_humans = r.get("is_humans")
        if is_humans is False:
            return True

    if exclude_pubtypes:
        pts = r.get("publication_types") or []
        pts_norm = {_norm_str(p) for p in pts}
        if pts_norm & DEFAULT_AUTOMATION_INELIGIBLE_PUBTYPES:
            return True

    return False


def _read_decisions_jsonl(path: Path) -> List[dict[str, Any]]:
    if not path.exists():
        return []
    out: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{line_no}: invalid JSON (not JSONL?)")
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _latest_decision_by_pmid_stage(decisions: List[dict[str, Any]]) -> dict[Tuple[str, str], dict[str, Any]]:
    latest: dict[Tuple[str, str], dict[str, Any]] = {}
    latest_ts: dict[Tuple[str, str], datetime] = {}

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
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = None

        if ts is None:
            latest[key] = obj
            continue

        prev = latest_ts.get(key)
        if prev is None or ts >= prev:
            latest_ts[key] = ts
            latest[key] = obj

    return latest



def _ensure_trailing_newline(path: Path) -> None:
    """Ensure file ends with newline so JSONL append can't corrupt the last line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fb:
        fb.seek(-1, 2)
        last = fb.read(1)
    if last != b"\n":
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")


# Helper: Write raw model output for debugging failed parses
def _write_debug_raw(debug_dir: Path, pmid: str, tag: str, content: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_pmid = re.sub(r"[^0-9A-Za-z._-]+", "_", pmid)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = debug_dir / f"{safe_pmid}.{tag}.{ts}.txt"
    out.write_text(content or "", encoding="utf-8")


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from model output.

    We ask for JSON-only output, but models sometimes wrap it in prose, code fences,
    or emit Python dicts / trailing commas. We keep fixes conservative.
    """
    if not text:
        return None

    s = text.strip()

    # Fast path: whole output is a JSON object
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Find first balanced {...} block (more reliable than greedy regex)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    end = None
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    chunk = s[start:end].strip()

    # Attempt strict JSON
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Conservative normalization
    norm = chunk
    norm = norm.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Remove trailing commas before } or ]
    norm = re.sub(r",\s*(\}|\])", r"\1", norm)

    # Quote common bareword enum mistakes (keep scope narrow)
    # e.g. "reason_code": WRONG_INTERVENTION  ->  "reason_code": "WRONG_INTERVENTION"
    norm = re.sub(
        r'("reason_code"\s*:\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[\},])',
        r'\1"\2"\3',
        norm,
    )
    norm = re.sub(
        r'("decision"\s*:\s*)(include|exclude|maybe)(\s*[\},])',
        r'\1"\2"\3',
        norm,
        flags=re.IGNORECASE,
    )
    norm = re.sub(
        r'("confidence"\s*:\s*)(low|medium|high)(\s*[\},])',
        r'\1"\2"\3',
        norm,
        flags=re.IGNORECASE,
    )

    try:
        obj = json.loads(norm)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Last resort: model may have emitted a Python dict
    try:
        obj2 = ast.literal_eval(norm)
        return obj2 if isinstance(obj2, dict) else None
    except Exception:
        return None


def _build_prompt(record: dict[str, Any], criteria_text: str) -> str:
    pmid = str(record.get("pmid") or "")
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    pubtypes = record.get("publication_types") or []
    languages = record.get("languages") or []
    is_humans = record.get("is_humans")

    return (
        "You are screening PubMed records for a systematic review with STRICT eligibility criteria.\n"
        "Use ONLY the provided Title/Abstract/metadata and the AUTHORITATIVE CRITERIA below.\n"
        "Do NOT infer missing details.\n"
        "If key information is missing, set decision=\"maybe\" and reason_code=\"UNCLEAR\".\n\n"
        "AUTHORITATIVE SCREENING CRITERIA (apply strictly):\n"
        + criteria_text
        + "\n"
        "Return EXACTLY one JSON object (no markdown, no commentary) with keys:\n"
        "Output MUST start with '{' and end with '}' and contain nothing else (single JSON object only).\n"
        "All string values MUST be in double quotes, including reason_code (e.g., \"reason_code\": \"WRONG_INTERVENTION\").\n"
        "pmid, decision, reason_code, confidence, notes\n"
        "Where decision is one of: include, exclude, maybe.\n"
        "reason_code MUST be consistent with decision:\n"
        "- if decision=exclude: choose one of NOT_RELEVANT, WRONG_DESIGN, WRONG_POPULATION, WRONG_OUTCOME, WRONG_INTERVENTION, NO_ABSTRACT, LANGUAGE, OTHER\n"
        "- if decision=include: use ELIGIBLE\n"
        "- if decision=maybe: use POTENTIALLY_ELIGIBLE or UNCLEAR\n"
        "confidence: low|medium|high\n"
        "notes: <= 20 words, strictly quoting or paraphrasing title/abstract (no speculation).\n\n"
        "Decision rules:\n"
        "- INCLUDE only if ALL eligibility criteria are explicitly satisfied.\n"
        "- EXCLUDE with WRONG_DESIGN if review/meta/protocol/editorial.\n"
        "- If uncertain due to missing details, choose MAYBE with UNCLEAR.\n\n"
        f"PMID: {pmid}\n"
        f"PublicationTypes: {pubtypes}\n"
        f"Languages: {languages}\n"
        f"HumansFlag: {is_humans}\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract if abstract else 'NO ABSTRACT'}\n"
    )



def _call_ollama_http(model: str, prompt: str, timeout_s: int = 180) -> str:
    """Call Ollama via local HTTP API and return the generated text.

    Uses POST http://localhost:11434/api/generate
    """
    url = "http://localhost:11434/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama HTTP request failed: {e}")

    if r.status_code != 200:
        # Ollama typically returns JSON with an 'error' field on failures.
        try:
            j = r.json()
            err = j.get("error") if isinstance(j, dict) else None
        except Exception:
            err = None
        msg = err or (r.text or "").strip() or f"HTTP {r.status_code}"
        raise RuntimeError(f"Ollama HTTP error: {msg}")

    try:
        data = r.json()
    except Exception:
        raise RuntimeError("Ollama HTTP response was not valid JSON")

    if not isinstance(data, dict):
        raise RuntimeError("Ollama HTTP response JSON had unexpected shape")

    out = data.get("response")
    if not isinstance(out, str):
        raise RuntimeError("Ollama HTTP response missing 'response' text")
    return out


def _call_ollama_subprocess(model: str, prompt: str, timeout_s: int = 180) -> str:
    """Call Ollama via CLI (subprocess) and return raw stdout.

    We pass the prompt as a CLI argument (not stdin) to avoid interactive/stdin quirks.
    """
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"ollama run failed (code {proc.returncode}): {err}")
    return proc.stdout


def _call_ollama(model: str, prompt: str, timeout_s: int = 180, transport: str = "http") -> str:
    """Call Ollama using the selected transport ('http' or 'subprocess')."""
    t = (transport or "http").strip().lower()
    if t == "subprocess":
        return _call_ollama_subprocess(model, prompt, timeout_s=timeout_s)
    return _call_ollama_http(model, prompt, timeout_s=timeout_s)


def _make_decision_line(pmid: str, stage: str, obj: dict[str, Any], by: str) -> Dict[str, Any]:
    decision = str(obj.get("decision") or "maybe").lower().strip()
    if decision not in {"include", "exclude", "maybe"}:
        decision = "maybe"

    rc = obj.get("reason_code")
    reason_code = str(rc).strip().upper() if rc is not None and str(rc).strip() else None

    # Normalize confidence
    conf = str(obj.get("confidence") or "").lower().strip()
    if conf not in {"low", "medium", "high"}:
        conf = None

    # Enforce decision/reason consistency (guard against model nonsense)
    if decision == "include":
        # If model provided an exclusion reason, flip to exclude.
        if reason_code in EXCLUSION_REASON_CODES or reason_code == "UNCLEAR":
            decision = "exclude"
        else:
            reason_code = "ELIGIBLE"

    if decision == "exclude":
        if reason_code not in EXCLUSION_REASON_CODES:
            # Default to OTHER if missing or invalid
            reason_code = "OTHER"

    if decision == "maybe":
        if reason_code is None:
            reason_code = "UNCLEAR"
        elif reason_code in EXCLUSION_REASON_CODES:
            # If they gave a clear exclusion reason, treat it as exclude.
            decision = "exclude"
        elif reason_code not in {"POTENTIALLY_ELIGIBLE", "UNCLEAR"}:
            reason_code = "UNCLEAR"

    # If the model outputs a different PMID, ignore it (we trust our input PMID).
    _pmid_out = obj.get("pmid")
    if _pmid_out and str(_pmid_out).strip() != pmid:
        # keep going; do not overwrite
        pass

    # Keep optional short notes
    notes = obj.get("notes")
    if notes is not None:
        notes = str(notes).strip()
        if len(notes) > 200:
            notes = notes[:200]

    line: Dict[str, Any] = {
        "pmid": pmid,
        "stage": stage,
        "decision": decision,
        "reason_code": reason_code,
        "by": by,
        "confidence": conf,
        "timestamp": _utc_now_iso(),
    }
    if notes:
        line["notes"] = notes
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-assisted screening for remaining 'maybe' records")
    ap.add_argument(
        "runs_dir",
        nargs="?",
        default="runs",
        help="Path to runs/ folder (default: ./runs)",
    )
    ap.add_argument(
        "--decisions",
        default=None,
        help="Decisions JSONL path (default: <runs_dir>/decisions.jsonl)",
    )
    ap.add_argument(
        "--stage",
        default="abstract",
        choices=["abstract", "fulltext"],
        help="Decision stage to write (default: abstract)",
    )
    ap.add_argument(
        "--criteria",
        default="criteriaImpuls.txt",
        help="Path to criteria text file to inject into the prompt (default: criteriaImpuls.txt)",
    )
    ap.add_argument(
        "--model",
        default="qwen3-coder-30bq8-screen:latest",
        help="Ollama model name (default: qwen3-coder-30bq8-screen:latest)",
    )
    ap.add_argument(
        "--transport",
        default="http",
        choices=["http", "subprocess"],
        help="How to call Ollama (default: http). Use 'subprocess' for CLI fallback.",
    )
    ap.add_argument(
        "--by",
        default="qwen",
        help="Value for the 'by' field (default: qwen)",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of PMIDs to process (0 = no limit)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not write anything; just show what would be processed",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout seconds per record (default: 180)",
    )
    ap.add_argument(
        "--debug-raw-dir",
        default=None,
        help="If set, write raw LLM outputs for any parse failures to this directory (default: <runs_dir>/llm_raw)",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    decisions_path = (
        Path(args.decisions).expanduser().resolve()
        if args.decisions
        else (runs_dir / "decisions.jsonl").resolve()
    )

    # Resolve criteria path (relative paths: script dir first, then current working dir)
    criteria_arg = Path(args.criteria).expanduser()
    if criteria_arg.is_absolute():
        criteria_path = criteria_arg
    else:
        script_dir = Path(__file__).resolve().parent
        candidate1 = (script_dir / criteria_arg).resolve()
        candidate2 = (Path.cwd() / criteria_arg).resolve()
        criteria_path = candidate1 if candidate1.exists() else candidate2

    criteria_text = _read_criteria_text(criteria_path)

    debug_raw_dir = (
        Path(args.debug_raw_dir).expanduser().resolve()
        if args.debug_raw_dir
        else (runs_dir / "llm_raw").resolve()
    )

    # Build screening universe (unique PMIDs post automation) and keep one representative record per PMID
    pmid_to_record: dict[str, dict[str, Any]] = {}
    for r in _iter_records(runs_dir):
        pmid = r.get("pmid")
        if not pmid:
            continue
        pmid_s = str(pmid)
        if pmid_s in pmid_to_record:
            continue
        if _automation_ineligible(r):
            continue
        pmid_to_record[pmid_s] = r

    decisions = _read_decisions_jsonl(decisions_path)
    latest = _latest_decision_by_pmid_stage(decisions)

    # Select candidates: missing or maybe
    candidates: List[str] = []
    for pmid in sorted(pmid_to_record.keys()):
        obj = latest.get((pmid, args.stage))
        current_dec = str((obj or {}).get("decision") or "missing").lower()
        if current_dec in {"missing", "maybe"}:
            candidates.append(pmid)

    if args.max and args.max > 0:
        candidates = candidates[: args.max]

    print(f"Runs dir: {runs_dir}")
    print(f"Decisions file: {decisions_path}")
    print(f"Screening universe (unique PMIDs, post-automation): {len(pmid_to_record)}")
    print(f"Candidates to send to LLM (missing/maybe): {len(candidates)}")
    print(f"Model: {args.model}")
    print(f"Transport: {args.transport}")
    print(f"Criteria file: {criteria_path}")

    if args.dry_run:
        if candidates:
            print("DRY-RUN PMIDs:")
            for pmid in candidates:
                print(f"  - {pmid}")
        return 0

    if not candidates:
        print("Nothing to do.")
        return 0

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(decisions_path)

    appended = 0
    errors = 0
    for pmid in candidates:
        r = pmid_to_record[pmid]
        prompt = _build_prompt(r, criteria_text)
        try:
            raw = _call_ollama(args.model, prompt, timeout_s=args.timeout, transport=args.transport)
            obj = _extract_first_json_object(raw)
            raw2 = ""
            if not obj:
                # One retry with a terse "repair" instruction
                repair = (
                    prompt
                    + "\n\nIMPORTANT: Your previous output was not valid JSON. "
                    + "Return ONLY one valid JSON object, starting with '{' and ending with '}', no extra text."
                )
                raw2 = _call_ollama(args.model, repair, timeout_s=args.timeout, transport=args.transport)
                obj = _extract_first_json_object(raw2)
            if not obj:
                _write_debug_raw(debug_raw_dir, pmid, "raw", raw)
                if raw2:
                    _write_debug_raw(debug_raw_dir, pmid, "raw_retry", raw2)
                raise ValueError(
                    f"Could not parse JSON from model output (saved raw to {debug_raw_dir})"
                )
            line = _make_decision_line(pmid, args.stage, obj, args.by)
            with decisions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            appended += 1
            print(f"OK {pmid}: {line['decision']} ({line.get('reason_code')})")
        except Exception as e:
            errors += 1
            print(f"ERROR {pmid}: {e}")
            print("  Hint: try running this PMID with --max 1 and/or increase --timeout; also ensure the model outputs JSON.")

    print(f"DONE. Appended: {appended}. Errors: {errors}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())