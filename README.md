# LLM-assisted evidence pipeline (PubMed → PRISMA → Screening → Extraction → Tables)

This repository contains a lightweight, local-first pipeline to:
1) retrieve records from PubMed,
2) generate PRISMA-style counts and intermediate sets,
3) perform LLM-assisted semantic screening,
4) extract structured fields from abstracts via an Ollama-served model (HTTP API),
5) generate descriptive evidence tables (CSV/Excel-like outputs via pandas).

The pipeline is designed for rapid, reproducible evidence mapping under constrained information (abstracts + metadata), with configurations fully versioned in JSON.

## Repository structure

- `scripts/`  
  Entry-point Python scripts (CLI).
- `configs/`  
  Versioned JSON configurations defining extraction schema and LLM prompts.
- `examples/`  
  Minimal example inputs/outputs to illustrate formats (records, decisions, extractions).
- `requirements.txt`  
  Minimal Python dependencies.

## Requirements

- Python 3.10+ recommended
- An Ollama server reachable via HTTP (for LLM steps)
- Network access for PubMed retrieval (if using the PubMed scripts)

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Use `-h` on every script to inspect available options:

```bash
python scripts/pubmed.py -h
python scripts/qwen_screen.py -h
python scripts/qwen_extract_abs.py -h
```

Typical end-to-end flow (see each script `--help` for required arguments):

1) Retrieve PubMed records (JSON):
```bash
python scripts/pubmed.py --help
```

2) PRISMA counts / intermediate sets:
```bash
python scripts/pubmed_prisma.py --help
python scripts/pubmed_decisions.py --help
python scripts/pubmed_included.py --help
```

3) LLM-assisted screening (Ollama via HTTP):
```bash
python scripts/qwen_screen.py --help
```

4) Structured extraction from abstracts (Ollama via HTTP) using a config:
```bash
python scripts/qwen_extract_abs.py --config configs/config_ketamine_extraction.json --help
```

5) Descriptive tables (pandas):
```bash
python scripts/qwen_descr_tablesKet.py --help
python scripts/qwen_descr_tables_clozsubj.py --help
python scripts/qwen_descr_tables_clozsui.py --help
python scripts/qwen_tables_Ket.py --help
```