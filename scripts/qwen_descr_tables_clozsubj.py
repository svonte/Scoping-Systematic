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
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print("[!] Skipping invalid JSON line")
                continue

            # 1. Extract Metadata (Descriptors) for clozapine-experience extraction fields
            row = {
                'First_Author': clean_cell(entry.get('first_author', 'NR')),
                'Year': clean_cell(entry.get('year', 'NR')),
                'PMID': clean_cell(entry.get('pmid', 'NR')),
                'Citation_Details': clean_cell(entry.get('citation_details', 'NR')),
                'Study_Design': clean_cell(entry.get('study_design', 'NR')),
                'Methodology_Assessment': clean_cell(entry.get('methodology_assessment', 'NR')),
                'Sample_Size': clean_cell(entry.get('sample_size', 'NR')),
                'Respondent_Type': clean_cell(entry.get('respondent_type', 'NR')),
                'Patient_Diagnosis': clean_cell(entry.get('patient_diagnosis', 'NR')),
                'Sociodemographics': clean_cell(entry.get('sociodemographics', 'NR')),
                'Treatment_Details': clean_cell(entry.get('treatment_details', 'NR')),
                'Experience_Dimensions': clean_cell(entry.get('experience_dimensions', 'NR')),
                'Key_Findings': clean_cell(entry.get('key_findings', 'NR')),
                'Notes_for_Synthesis': clean_cell(entry.get('notes_for_synthesis', 'NR'))
            }
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[*] Successfully created {output_file} with {len(df)} studies.")
    print("    Columns included:", ", ".join(df.columns))

if __name__ == "__main__":
    create_master_table("extracted_results.jsonl", "clozapine_experience_descriptive_table.csv")