import pandas as pd
import json

data = []
# Load your extracted data
with open('processed_results_v3.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        if 'structured_analysis' in entry and entry['structured_analysis']:
            for item in entry['structured_analysis']:
                # Flatten the data for the pivot table
                row = {
                    'Study': entry.get('authors_year'),
                    'Network': item.get('network'),
                    'Region': item.get('macro_region'),
                    'Modality': item.get('modality_type'),
                    'Context': item.get('finding_type'),
                    'Direction': item.get('direction')
                }
                data.append(row)

df = pd.DataFrame(data)

# --- Table 1: Biological Consensus (Drug Effects Only) ---
# Filter for general drug effects (ignoring baseline predictors)
df_bio = df[df['Context'] == 'Drug_Effect_General']

# Calculate the "Mode" (most common finding) for each Network + Modality
table1 = df_bio.pivot_table(
    index='Network', 
    columns='Modality', 
    values='Direction', 
    aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 'NR'
)

print("\n--- Table 1: Biological Consensus ---")
print(table1)

# --- Table 2: Clinical Biomarkers ---
# Filter for Baseline Predictors and Correlates
df_clin = df[df['Context'].isin(['Baseline_Predictor', 'Correlate_of_Improvement'])]

# Count how many studies support each finding
table2 = df_clin.groupby(['Region', 'Context', 'Direction']).size().reset_index(name='Study_Count')
table2 = table2.sort_values('Study_Count', ascending=False)

print("\n--- Table 2: Top Clinical Signals ---")
print(table2.head(10))

# Save for plotting
table1.to_csv("table1_bio_consensus.csv")
table2.to_csv("table2_clinical_signals.csv")