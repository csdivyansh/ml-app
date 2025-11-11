#!/usr/bin/env python3
"""
Generate a cleaned dataset and symptom normalization map from DiseaseAndSymptoms.csv.
Outputs:
 - ml-api/cleaned_disease_symptoms.csv  (columns: Disease,Symptoms)
 - ml-api/symptom_map.json              (original_token -> normalized_token)
 - ml-api/clean_dataset_stats.json      (row counts, unique counts, averages)

Run: python ml-api/generate_clean_dataset.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "DiseaseAndSymptoms.csv"
OUT_CSV = ROOT / "cleaned_disease_symptoms.csv"
SYM_MAP = ROOT / "symptom_map.json"
STATS = ROOT / "clean_dataset_stats.json"


def normalize(tok: str):
    if tok is None:
        return None
    t = str(tok).strip()
    if t == '':
        return None
    # remove trailing commas/periods and extra whitespace
    t = t.strip(' ,.')
    t = t.lower()
    # unify stray spaces around underscores
    t = re.sub(r"\s*_\s*", "_", t)
    # collapse whitespace then replace with underscore
    t = re.sub(r"\s+", " ", t)
    t = t.replace(' ', '_')
    t = re.sub(r"_+", "_", t)
    if t in ('nan', ''):
        return None
    return t


def main():
    if not INPUT.exists():
        print("Input file not found:", INPUT)
        return

    total_rows = 0
    unique_diseases = Counter()
    orig_to_norm = {}
    norm_counter = Counter()
    symptoms_per_row = []

    out_rows = []

    with INPUT.open(newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise SystemExit('empty file')
        # assume first column Disease, rest are symptom columns
        symptom_cols = header[1:]

        for row in reader:
            # skip fully-empty rows
            if not any(cell.strip() for cell in row):
                continue
            total_rows += 1
            disease = row[0].strip()
            unique_diseases[disease] += 1

            tokens = []
            for cell in row[1:]:
                if cell is None:
                    continue
                raw = str(cell).strip()
                if raw == '':
                    continue
                # sometimes a symptom cell may contain multiple comma-separated tokens
                # split on comma if present
                parts = [p.strip() for p in raw.split(',') if p.strip()]
                for p in parts:
                    norm = normalize(p)
                    if norm:
                        tokens.append((p, norm))

            # dedupe preserving order on normalized token
            seen = set()
            deduped_norm = []
            for orig, norm in tokens:
                if norm not in seen:
                    deduped_norm.append(norm)
                    seen.add(norm)
                # record mapping from original -> normalized (prefer existing mapping)
                if orig not in orig_to_norm:
                    orig_to_norm[orig] = norm

            for n in deduped_norm:
                norm_counter[n] += 1
            symptoms_per_row.append(len(deduped_norm))

            out_rows.append((disease, ';'.join(deduped_norm)))

    # write cleaned csv
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Disease', 'Symptoms'])
        for disease, symptoms in out_rows:
            writer.writerow([disease, symptoms])

    # write symptom map (original -> normalized)
    with SYM_MAP.open('w', encoding='utf-8') as f:
        json.dump(orig_to_norm, f, indent=2, ensure_ascii=False)

    stats = {
        'rows': total_rows,
        'symptom_columns_detected': len(symptom_cols),
        'unique_diseases': len(unique_diseases),
        'unique_symptoms_normalized': len(norm_counter),
        'avg_symptoms_per_row': round(sum(symptoms_per_row)/len(symptoms_per_row) if symptoms_per_row else 0, 2),
        'top_30_symptoms': norm_counter.most_common(30),
        'top_20_diseases': unique_diseases.most_common(20),
    }

    with STATS.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote cleaned CSV: {OUT_CSV}")
    print(f"Wrote symptom map: {SYM_MAP}")
    print(f"Wrote stats: {STATS}")


if __name__ == '__main__':
    main()
