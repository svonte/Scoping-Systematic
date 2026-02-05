import argparse
import json
import requests
import re
from pathlib import Path
from datetime import datetime, timezone

def extract_json(text: str):
    """Robust extraction of JSON block from raw text."""
    # Try to find content between { and }
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def run_extraction(config_path, input_path, out_path, max_records=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    fields_schema = json.dumps(config['extraction_fields'], indent=2)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    records = data.get("records", [])

    processed_pmids = set()
    if out_path.exists():
        with open(out_path, 'r') as f:
            for line in f:
                try:
                    processed_pmids.add(json.loads(line).get("pmid"))
                except: continue

    print(f"Engine started. Model: {config['model_name']}")

    count = 0
    for rec in records:
        if max_records and count >= max_records: break
        pmid = rec.get("pmid")
        if pmid in processed_pmids: continue

        # Prompt pulito ed efficace
        system_content = (
            "You are an information extraction engine.\n"
            "Task: Extract clinical trial data into ONE valid JSON object using EXACTLY the keys in the schema below.\n\n"
            f"SCHEMA (keys + descriptions):\n{fields_schema}\n\n"
            "HARD CONSTRAINTS (must follow):\n"
            "1) Use ONLY the provided Title/Authors/Year/Abstract text. Do NOT use outside knowledge.\n"
            "2) Do NOT guess or infer missing values. If something is not explicitly stated, output 'NR'.\n"
            "3) Do NOT invent sample sizes, p-values, effect sizes, timepoints, or claim significance/non-significance unless explicitly stated.\n"
            "4) For results_summary_complete: summarize ONLY outcomes explicitly reported, and include arms + timepoints if explicitly stated; otherwise 'NR' for those elements.\n"
            "5) Output MUST be raw JSON only (no markdown, no commentary, no extra keys).\n"
            "6) Before finalizing, internally verify each extracted value is explicitly supported by the text; otherwise set it to 'NR'.\n"
        )
        authors = ", ".join(rec.get("authors", []))
        user_content = f"Title: {rec.get('title')}\nAuthors: {authors}\nYear: {rec.get('year')}\nAbstract: {rec.get('abstract')}"

        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            # NOTA: format: json rimosso per stabilità
            "options": {"temperature": 0.01, "num_ctx": 4096}
        }

        try:
            response = requests.post(config["ollama_url"], json=payload, timeout=360)
            response.raise_for_status()
            
            raw_content = response.json().get("message", {}).get("content", "").strip()
            extracted_data = extract_json(raw_content)

            if extracted_data:
                extracted_data["pmid"] = pmid
                extracted_data["_timestamp"] = datetime.now(timezone.utc).isoformat()
                
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(extracted_data, ensure_ascii=False) + "\n")
                
                print(f"[+] Processed: {pmid}")
                count += 1
            else:
                print(f"[-] Failed to parse JSON for {pmid}. Raw: {raw_content[:50]}...")

        except Exception as e:
            print(f"[!] Network error for {pmid}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input", default="included_records.json")
    parser.add_argument("--out", default="extracted_results.jsonl")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    run_extraction(args.config, Path(args.input), Path(args.out), args.max)