# Cleaning dataset

This folder contains a small helper to normalize and clean the symptom dataset.

Files created:

- `generate_clean_dataset.py` - script that reads `DiseaseAndSymptoms.csv` and writes:
  - `cleaned_disease_symptoms.csv` (columns: `Disease`, `Symptoms` where `Symptoms` is a semicolon-separated list of normalized tokens)
  - `symptom_map.json` (mapping of observed original token -> normalized token)
  - `clean_dataset_stats.json` (summary statistics)

How to run locally (no heavy dependencies required):

```bash
python ml-api/generate_clean_dataset.py
```

The script performs token normalization (lowercase, trim, collapse spaces, unify underscores), deduplicates symptoms per row, and preserves order.

Next steps:
- Use `cleaned_disease_symptoms.csv` to train models easily: split symptoms by `;` into lists and use `MultiLabelBinarizer`.
- Inspect `symptom_map.json` for normalization fixes (e.g. synonyms) and optionally extend the map manually.
