import json
from pathlib import Path
from datetime import datetime

runs = Path("runs")
stage = "abstract"

# 1) Loads records
records = {}
for fp in runs.rglob("records.json"):
    data = json.load(fp.open())
    for r in data["records"]:
        pmid = str(r.get("pmid"))
        if pmid:
            records[pmid] = r

# 2) Load decisions
latest = {}
with (runs / "decisions.jsonl").open() as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("stage") != stage:
            continue
        pmid = str(obj.get("pmid"))
        ts = datetime.fromisoformat(obj["timestamp"].replace("Z","+00:00"))
        key = pmid
        if key not in latest or ts > latest[key]["ts"]:
            latest[key] = {"ts": ts, "decision": obj["decision"]}

# 3) Filters included
included = [
    records[pmid]
    for pmid, d in latest.items()
    if d["decision"] == "include" and pmid in records
]

# 4) Save
out = {
    "n_included": len(included),
    "records": included,
}

Path("included_records.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False)
)

print("Included:", len(included))