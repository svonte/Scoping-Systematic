#!/usr/bin/env python3
"""pubmed1.py

Minimal, reproducible PubMed (NCBI E-utilities) search + abstract retrieval.

What it does (by design):
- Runs one PubMed query and retrieves up to N records (default 5000), sorted by most recent first.
- Fetches metadata + abstract (when available) for those PMIDs.
- Saves a single JSON file with structured records.

Notes:
- PubMed changes over time; this script intentionally does NOT try to freeze results.
- For reproducibility, it stores the query string and retrieval timestamp in the output.

Usage examples:
  python pubmed1.py --query 'major depressive disorder polygenic score treatment response' --max-results 5000 --out pubmed_results.json
  NCBI_EMAIL='you@uni.it' python pubmed1.py --query 'schizophrenia inflammation CRP' --max-results 50

Optional env vars:
- NCBI_EMAIL   (recommended by NCBI)
- NCBI_API_KEY (optional; increases rate limit)
- NCBI_TOOL    (optional; tool name)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# EFetch batch size when using the Entrez history server (WebEnv/query_key).
# 200 is a conservative default that keeps responses small and resilient.
EFETCH_BATCH_SIZE = 200


@dataclass
class PubMedSearchResult:
    pmids: List[str]
    count: int
    webenv: Optional[str]
    query_key: Optional[str]

@dataclass
class PubMedRecord:
    pmid: str
    title: Optional[str]
    abstract: Optional[str]
    authors: List[str]
    journal: Optional[str]
    year: Optional[int]

    # Screening-relevant PubMed metadata
    publication_types: List[str]
    languages: List[str]
    mesh_terms: List[str]
    is_humans: Optional[bool]
    doi: Optional[str]
    pmcid: Optional[str]


def _ncbi_common_params() -> Dict[str, str]:
    """Common parameters recommended by NCBI."""
    params: Dict[str, str] = {}
    email = os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")
    tool = os.environ.get("NCBI_TOOL", "revpy26")

    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    if tool:
        params["tool"] = tool
    return params


def _rate_limit_sleep() -> None:
    """Sleep to respect NCBI E-utilities rate limits.

    NCBI guidance (typical):
    - Without API key: <= 3 requests/second
    - With API key: <= 10 requests/second

    We use conservative sleeps.
    """
    has_key = bool(os.environ.get("NCBI_API_KEY"))
    time.sleep(0.12 if has_key else 0.40)


def pubmed_esearch(
    query: str,
    max_results: int = 5000,
    sort: str = "date",
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
) -> PubMedSearchResult:
    """Search PubMed and return history-server handles (WebEnv/query_key).

    IMPORTANT: For PubMed, ESearch cannot page idlists beyond the first 9,999 records.
    Therefore we always call ESearch with retmax=0 + usehistory=y and later partition by date
    when we need to retrieve more than 9,999 records.

    Note: `max_results` is an upper bound on how many records we will later fetch.
    """
    if not query.strip():
        raise ValueError("Query is empty.")
    if max_results <= 0:
        return PubMedSearchResult(pmids=[], count=0, webenv=None, query_key=None)

    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 0,
        "sort": sort,
        "usehistory": "y",
    }

    if mindate or maxdate:
        params["datetype"] = "pdat"
        if mindate:
            params["mindate"] = mindate
        if maxdate:
            params["maxdate"] = maxdate

    params.update(_ncbi_common_params())

    r = requests.get(ESEARCH_URL, params=params, timeout=60)
    _rate_limit_sleep()
    r.raise_for_status()

    # PubMed sometimes returns JSON with unescaped control characters; sanitize if needed.
    try:
        data = r.json()
    except ValueError:
        txt = r.text
        txt = txt.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        data = json.loads(txt)

    esr = data.get("esearchresult", {})

    # If PubMed returns an error field, fail fast with a readable message.
    if isinstance(esr, dict) and esr.get("ERROR"):
        raise RuntimeError(str(esr.get("ERROR")))

    count_str = str(esr.get("count", "0")).strip()
    count = int(count_str) if count_str.isdigit() else 0

    webenv = esr.get("webenv")
    query_key = esr.get("querykey")

    return PubMedSearchResult(
        pmids=[],
        count=count,
        webenv=str(webenv) if webenv else None,
        query_key=str(query_key) if query_key else None,
    )

def _parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")


def _fmt_ymd(d: datetime) -> str:
    return d.strftime("%Y/%m/%d")


def _today_ymd() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y/%m/%d")


def _partition_date_ranges(
    query: str,
    sort: str,
    mindate: str,
    maxdate: str,
    max_per_partition: int = 9999,
) -> List[Tuple[str, str, PubMedSearchResult]]:
    """Recursively split [mindate, maxdate] until each partition has <= max_per_partition hits."""
    res = pubmed_esearch(query=query, max_results=1, sort=sort, mindate=mindate, maxdate=maxdate)
    if res.count <= max_per_partition:
        if not res.webenv or not res.query_key:
            raise RuntimeError("ESearch did not return history server handles (WebEnv/query_key).")
        return [(mindate, maxdate, res)]

    start = _parse_ymd(mindate)
    end = _parse_ymd(maxdate)
    if start >= end:
        raise RuntimeError(f"Cannot split range further: {mindate}..{maxdate} still has {res.count} hits")

    mid = start + (end - start) / 2
    mid = datetime(mid.year, mid.month, mid.day)

    left_max = _fmt_ymd(mid)
    right_min = _fmt_ymd(mid + timedelta(days=1))

    if left_max == mindate and right_min == maxdate:
        raise RuntimeError(f"Date partitioning stalled at {mindate}..{maxdate} with {res.count} hits")

    left = _partition_date_ranges(query, sort, mindate, left_max, max_per_partition)
    right = _partition_date_ranges(query, sort, right_min, maxdate, max_per_partition)
    return left + right

def pubmed_efetch_xml(pmids: List[str]) -> str:
    """Fetch PubMed records as XML for a list of PMIDs."""
    if not pmids:
        return ""

    params: Dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    params.update(_ncbi_common_params())

    r = requests.get(EFETCH_URL, params=params, timeout=120)
    _rate_limit_sleep()
    r.raise_for_status()
    return r.text


def pubmed_efetch_xml_history(
    webenv: str,
    query_key: str,
    retstart: int,
    retmax: int,
) -> str:
    """Fetch PubMed records as XML using Entrez history server (WebEnv/query_key)."""
    if retmax <= 0:
        return ""

    params: Dict[str, Any] = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": int(retstart),
        "retmax": int(retmax),
        "retmode": "xml",
    }
    params.update(_ncbi_common_params())

    r = requests.get(EFETCH_URL, params=params, timeout=120)
    _rate_limit_sleep()
    r.raise_for_status()
    return r.text


def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    txt = (elem.text or "").strip()
    return txt or None


def _join_text_with_children(elem: Optional[ET.Element]) -> Optional[str]:
    """Join text of an element including its children (for AbstractText with labels, italics, etc.)."""
    if elem is None:
        return None

    parts: List[str] = []

    def _recurse(e: ET.Element) -> None:
        if e.text and e.text.strip():
            parts.append(e.text.strip())
        for c in list(e):
            _recurse(c)
            if c.tail and c.tail.strip():
                parts.append(c.tail.strip())

    _recurse(elem)
    joined = " ".join(parts).strip()
    return joined or None


def parse_pubmed_xml(xml_text: str) -> List[PubMedRecord]:
    """Parse PubMedArticleSet XML into structured records."""
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    records: List[PubMedRecord] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _text_or_none(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            # Skip malformed entries
            continue

        title = _join_text_with_children(article.find(".//Article/ArticleTitle"))

        # Abstract can be split into multiple AbstractText blocks.
        abstract_texts = article.findall(".//Article/Abstract/AbstractText")
        if abstract_texts:
            chunks: List[str] = []
            for a in abstract_texts:
                label = a.attrib.get("Label") or a.attrib.get("NlmCategory")
                body = _join_text_with_children(a)
                if not body:
                    continue
                if label:
                    chunks.append(f"{label}: {body}")
                else:
                    chunks.append(body)
            abstract = "\n".join(chunks).strip() or None
        else:
            abstract = None

        # Authors
        authors: List[str] = []
        for au in article.findall(".//Article/AuthorList/Author"):
            last = _text_or_none(au.find("LastName"))
            fore = _text_or_none(au.find("ForeName"))
            coll = _text_or_none(au.find("CollectiveName"))

            if coll:
                authors.append(coll)
                continue
            if last and fore:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)

        journal = _text_or_none(article.find(".//Article/Journal/Title"))

        # Year: prefer PubDate/Year, fallback to MedlineDate parsing.
        year: Optional[int] = None
        year_txt = _text_or_none(article.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if year_txt and year_txt.isdigit():
            year = int(year_txt)
        else:
            medline_date = _text_or_none(article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
            if medline_date:
                # Extract first 4-digit year if present
                for token in medline_date.replace(";", " ").replace("-", " ").split():
                    if len(token) == 4 and token.isdigit():
                        year = int(token)
                        break

        # Publication types (useful for screening: RCT, Review, Meta-Analysis, etc.)
        publication_types: List[str] = []
        for pt in article.findall(".//Article/PublicationTypeList/PublicationType"):
            t = _text_or_none(pt)
            if t:
                publication_types.append(t)

        # Languages (often a short code like 'eng')
        languages: List[str] = []
        for lang in article.findall(".//Article/Language"):
            t = _text_or_none(lang)
            if t:
                languages.append(t)

        # MeSH terms and a simple Humans flag (when MeSH is present)
        mesh_terms: List[str] = []
        mesh_lower: List[str] = []
        for mh in article.findall(".//MedlineCitation/MeshHeadingList/MeshHeading"):
            desc = mh.find("DescriptorName")
            t = _text_or_none(desc)
            if t:
                mesh_terms.append(t)
                mesh_lower.append(t.lower())

        is_humans: Optional[bool] = None
        if mesh_lower:
            # Only set if MeSH exists; otherwise keep None (unknown)
            is_humans = "humans" in mesh_lower

        # Article identifiers (DOI / PMCID) when available
        doi: Optional[str] = None
        pmcid: Optional[str] = None
        for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = (aid.attrib.get("IdType") or "").lower().strip()
            val = _text_or_none(aid)
            if not val:
                continue
            if id_type == "doi" and doi is None:
                doi = val
            elif id_type == "pmc" and pmcid is None:
                pmcid = val

        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                publication_types=publication_types,
                languages=languages,
                mesh_terms=mesh_terms,
                is_humans=is_humans,
                doi=doi,
                pmcid=pmcid,
            )
        )

    return records


def retrieve_pubmed(
    query: str,
    max_results: int = 5000,
    sort: str = "date",
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
) -> Tuple[List[str], List[PubMedRecord], int]:
    """Retrieve up to max_results PubMed records for the query."""
    base = pubmed_esearch(query=query, max_results=max_results, sort=sort, mindate=mindate, maxdate=maxdate)

    total_available = int(base.count)
    target_n = min(int(max_results), total_available) if total_available > 0 else int(max_results)

    all_records: List[PubMedRecord] = []
    seen_pmids: set[str] = set()

    def _fetch_history_partition(webenv: str, query_key: str, n: int, label: str = "") -> None:
        retstart = 0
        while retstart < n:
            batch_n = min(EFETCH_BATCH_SIZE, n - retstart)
            print(
                f"[info] Fetching PubMed records {retstart + 1}–{retstart + batch_n} of {n}{label}",
                file=sys.stderr,
            )
            xml_text = pubmed_efetch_xml_history(
                webenv=webenv,
                query_key=query_key,
                retstart=retstart,
                retmax=batch_n,
            )
            batch_records = parse_pubmed_xml(xml_text)
            for r in batch_records:
                if r.pmid in seen_pmids:
                    continue
                seen_pmids.add(r.pmid)
                all_records.append(r)
            retstart += batch_n

    if target_n <= 9999:
        if not base.webenv or not base.query_key:
            raise RuntimeError("ESearch did not return history server handles (WebEnv/query_key).")
        _fetch_history_partition(base.webenv, base.query_key, target_n)
    else:
        md = mindate or "1970/01/01"
        xd = maxdate or _today_ymd()

        parts = _partition_date_ranges(query=query, sort=sort, mindate=md, maxdate=xd, max_per_partition=9999)

        for p_md, p_xd, p_res in parts:
            if len(all_records) >= target_n:
                break
            remain = target_n - len(all_records)
            n_part = min(remain, int(p_res.count))
            label = f" [{p_md}..{p_xd}]"
            _fetch_history_partition(p_res.webenv, p_res.query_key, n_part, label)

    pmids = [r.pmid for r in all_records]
    return pmids, all_records, total_available


def build_output_payload(
    query: str,
    pmids: List[str],
    records: List[PubMedRecord],
    total_hits: int,
    target_n: int,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).astimezone().isoformat()
    payload: Dict[str, Any] = {
        "esearch_count_total": total_hits,
        "target_n": target_n,
        "query": query,
        "retrieved_at": now,
        "count_pmids": len(pmids),
        "count_records": len(records),
        "pmids": pmids,
        "records": [asdict(r) for r in records],
        "notes": {
            "pubmed_dynamic": "PubMed is updated continuously; results may change over time.",
            "abstract_missing": "Some PubMed records do not include an abstract.",
        },
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Search PubMed and retrieve abstracts + metadata into one JSON file.")
    p.add_argument("--query", default=None, help="PubMed query string (as you would type in PubMed).")
    p.add_argument("--query-file", default=None, help="Path to a text file containing the PubMed query.")
    p.add_argument("--max-results", type=int, default=5000, help="Maximum number of records to retrieve (default: 5000).")
    p.add_argument("--sort", default="date", choices=["relevance", "date"], help="Sort order (default: date = most recent first).")
    p.add_argument("--mindate", default=None, help="Optional min publication date (YYYY/MM/DD).")
    p.add_argument("--maxdate", default=None, help="Optional max publication date (YYYY/MM/DD).")
    p.add_argument("--out", default="pubmed_results.json", help="Output JSON file path.")

    args = p.parse_args(argv)

    # Resolve query from file (preferred) or CLI
    query: Optional[str] = None

    if args.query_file:
        qpath = Path(args.query_file)
        if not qpath.exists():
            print(f"[error] query-file not found: {qpath}", file=sys.stderr)
            return 1
        query = qpath.read_text(encoding="utf-8").strip()
    elif args.query:
        query = str(args.query).strip()

    if not query:
        print("[error] Provide --query or --query-file (non-empty).", file=sys.stderr)
        return 1

    try:
        pmids, records, total_hits = retrieve_pubmed(
            query=query,
            max_results=args.max_results,
            sort=args.sort,
            mindate=args.mindate,
            maxdate=args.maxdate,
        )
    except requests.HTTPError as e:
        print(f"[error] HTTP error: {e}", file=sys.stderr)
        return 2
    except requests.RequestException as e:
        print(f"[error] Request failed: {e}", file=sys.stderr)
        return 2
    except ET.ParseError as e:
        print(f"[error] Failed to parse PubMed XML: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[error] Unexpected error: {e}", file=sys.stderr)
        return 1

    payload = build_output_payload(
        query,
        pmids,
        records,
        total_hits=total_hits,
        target_n=len(pmids),
    )

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to: {out_path}")
    if len(records) < len(pmids):
        print(f"[warn] Retrieved {len(pmids)} PMIDs but parsed {len(records)} records (some records may be missing/malformed).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())