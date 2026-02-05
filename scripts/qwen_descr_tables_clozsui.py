import pandas as pd
import json

def clean_cell(value):
    """Make values CSV-friendly: stringify, strip, and collapse whitespace/newlines."""
    if value is None:
        return "NR"
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    s = str(value).strip()
    if not s:
        return "NR"
    # Collapse all whitespace (including newlines/tabs) into single spaces
    s = " ".join(s.split())
    return s

def create_master_table(input_file, output_file):
    print(f"[*] Reading from {input_file}...")
    
    rows = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            entry = json.loads(line)
            
            # 1. Extract Metadata (Descriptors) for clozapine-suicidality extraction fields
            row = {
                'First_Author': clean_cell(entry.get('first_author', 'NR')),
                'Year': clean_cell(entry.get('year', 'NR')),
                'PMID': clean_cell(entry.get('pmid', 'NR')),
                'Type_of_Study': clean_cell(entry.get('type_of_study', 'NR')),
                'Aim': clean_cell(entry.get('aim', 'NR')),
                'Sample_Size': clean_cell(entry.get('sample_size', 'NR')),
                'Diagnosis': clean_cell(entry.get('diagnosis', 'NR')),
                'Clozapine_Dosage_Duration': clean_cell(entry.get('clozapine_dosages_duration', 'NR')),
                'Outcome': clean_cell(entry.get('outcome', 'NR')),
                'Summary_of_Results': clean_cell(entry.get('summary_of_results', 'NR')),
                'Effect_Direction': clean_cell(entry.get('effect_direction', 'NR')),
                'Effect_Basis': clean_cell(entry.get('effect_basis', 'NR'))
            }
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[*] Successfully created {output_file} with {len(df)} studies.")
    print("    Columns included:", ", ".join(df.columns))

if __name__ == "__main__":
    create_master_table("extracted_results.jsonl", "clozapine_suicidality_descriptive_table.csv")