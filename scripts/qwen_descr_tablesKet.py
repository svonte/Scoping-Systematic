import pandas as pd
import json

def format_findings(analysis_list):
    """
    Converts the list of structured findings into a single readable string.
    Format: [Network] Region (Modality): Direction
    """
    if not analysis_list:
        return "NR"
    
    formatted_items = []
    for item in analysis_list:
        # Create a compact string for each finding
        # e.g., "SN (Cingulate): Increased [Correlate]"
        
        network = item.get('network', 'NR')
        region = item.get('macro_region', 'NR')
        modality = item.get('modality_type', 'NR')
        direction = item.get('direction', 'NR')
        context = item.get('finding_type', 'General')
        
        # Shorten context for readability
        if context == "Correlate_of_Improvement":
            context_tag = "[Corr]"
        elif context == "Baseline_Predictor":
            context_tag = "[Pred]"
        else:
            context_tag = "[Drug]"

        summary = f"{network} ({region}): {direction} {context_tag}"
        formatted_items.append(summary)
    
    return "; ".join(formatted_items)

def create_master_table(input_file, output_file):
    print(f"[*] Reading from {input_file}...")
    
    rows = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            entry = json.loads(line)
            
            # 1. Extract Metadata (Descriptors)
            row = {
                'Author_Year': entry.get('authors_year', 'NR'),
                'PMID': entry.get('pmid', 'NR'),
                'N_TRD': entry.get('n_trd', 'NR'),
                'N_HC': entry.get('n_hc', 'NR'),
                'Age_TRD': entry.get('age_trd', 'NR'),
                'Sex_TRD': entry.get('sex_trd', 'NR'),
                'Route': entry.get('ketamine_route', 'NR'),
                'Dosing': entry.get('ketamine_dosing', 'NR'),
                'MRI_Modality': entry.get('mri_modality', 'NR'),
                'Analysis_Method': entry.get('mri_analysis_task', 'NR'),
                'Response_Definition': entry.get('outcome_definition', 'NR'),
                
                # 2. Extract Findings
                # Structured (Cleaned by LLM)
                'Structured_Findings': format_findings(entry.get('structured_analysis', [])),
                # Raw Text (Original Abstract for reference)
                'Raw_Main_Findings': entry.get('main_findings', 'NR')
            }
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[*] Successfully created {output_file} with {len(df)} studies.")
    print("    Columns included:", ", ".join(df.columns))

if __name__ == "__main__":
    create_master_table("processed_results_v2.jsonl", "master_summary_table.csv")