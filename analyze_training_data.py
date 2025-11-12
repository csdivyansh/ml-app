"""
Analyze training data to understand why confidence is low for certain predictions.
Examines disease distribution, symptom overlap, and class balance.
"""
import os
import pandas as pd
import numpy as np
from collections import Counter
import json

BASE = os.path.dirname(__file__)
SYM_CSV = os.path.join(BASE, "DiseaseAndSymptoms.csv")


def analyze_training_data():
    print("=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    # Load dataset
    sym_df = pd.read_csv(SYM_CSV)
    print(f"\n📊 Dataset Shape: {sym_df.shape}")
    print(f"   Total rows: {len(sym_df)}")
    print(f"   Total columns: {len(sym_df.columns)}")
    
    # Get symptom columns
    symptom_cols = [col for col in sym_df.columns if col.startswith("Symptom")]
    print(f"\n💊 Symptom Columns: {len(symptom_cols)}")
    
    # Normalize symptoms
    for col in symptom_cols:
        sym_df[col] = sym_df[col].str.lower().str.strip()
    
    # Merge symptoms into list
    sym_df["symptoms"] = sym_df[symptom_cols].apply(
        lambda row: [s for s in row if pd.notna(s) and s != ""], axis=1
    )
    
    # Disease distribution
    print("\n" + "=" * 70)
    print("1. DISEASE DISTRIBUTION (Class Balance)")
    print("=" * 70)
    disease_counts = sym_df['Disease'].value_counts()
    print(f"\nTotal unique diseases: {len(disease_counts)}")
    print(f"Min samples per disease: {disease_counts.min()}")
    print(f"Max samples per disease: {disease_counts.max()}")
    print(f"Mean samples per disease: {disease_counts.mean():.2f}")
    print(f"Median samples per disease: {disease_counts.median():.0f}")
    
    print("\n⚠️ Diseases with only 1 sample (most likely low confidence):")
    single_sample = disease_counts[disease_counts == 1]
    for disease, count in single_sample.items():
        print(f"   - {disease}: {count} sample")
    
    print(f"\n✅ Top 10 diseases by sample count:")
    for disease, count in disease_counts.head(10).items():
        print(f"   - {disease}: {count} samples")
    
    # Check specific diseases
    print("\n" + "=" * 70)
    print("2. SPECIFIC DISEASE ANALYSIS")
    print("=" * 70)
    
    target_diseases = ['AIDS', 'Bronchial Asthma', 'GERD', 'Impetigo', 'Urinary tract infection']
    for disease in target_diseases:
        count = disease_counts.get(disease, 0)
        if count > 0:
            disease_data = sym_df[sym_df['Disease'] == disease]
            symptoms = disease_data['symptoms'].iloc[0]
            print(f"\n🔍 {disease}:")
            print(f"   Samples: {count}")
            print(f"   Symptoms: {symptoms}")
        else:
            print(f"\n❌ {disease}: NOT FOUND in dataset")
    
    # Symptom statistics
    print("\n" + "=" * 70)
    print("3. SYMPTOM STATISTICS")
    print("=" * 70)
    
    all_symptoms = []
    for symptom_list in sym_df['symptoms']:
        all_symptoms.extend(symptom_list)
    
    symptom_counts = Counter(all_symptoms)
    print(f"\nTotal unique symptoms: {len(symptom_counts)}")
    print(f"Total symptom occurrences: {len(all_symptoms)}")
    
    print(f"\n✅ Top 10 most common symptoms:")
    for symptom, count in symptom_counts.most_common(10):
        print(f"   - {symptom}: {count} occurrences")
    
    # Check specific symptoms
    print("\n🔍 Symptoms from test case (high_fever, cough):")
    for symptom in ['high_fever', 'cough']:
        count = symptom_counts.get(symptom, 0)
        if count > 0:
            # Find diseases with this symptom
            diseases_with_symptom = []
            for _, row in sym_df.iterrows():
                if symptom in row['symptoms']:
                    diseases_with_symptom.append(row['Disease'])
            
            print(f"\n   {symptom}:")
            print(f"      Occurrences: {count}")
            print(f"      Found in diseases: {diseases_with_symptom[:10]}")  # Show first 10
        else:
            print(f"\n   {symptom}: NOT FOUND")
    
    # Symptom overlap analysis
    print("\n" + "=" * 70)
    print("4. SYMPTOM OVERLAP ANALYSIS")
    print("=" * 70)
    
    avg_symptoms_per_disease = sym_df['symptoms'].apply(len).mean()
    print(f"\nAverage symptoms per disease: {avg_symptoms_per_disease:.2f}")
    
    # Find diseases with high_fever + cough
    print("\n🔍 Diseases with both 'high_fever' AND 'cough':")
    matching_diseases = []
    for _, row in sym_df.iterrows():
        symptoms = row['symptoms']
        if 'high_fever' in symptoms and 'cough' in symptoms:
            matching_diseases.append(row['Disease'])
    
    if matching_diseases:
        print(f"   Found {len(matching_diseases)} diseases:")
        for disease in matching_diseases[:15]:  # Show first 15
            print(f"   - {disease}")
    else:
        print("   None found with exact match")
    
    # Calculate symptom similarity for AIDS
    print("\n" + "=" * 70)
    print("5. WHY LOW CONFIDENCE FOR AIDS?")
    print("=" * 70)
    
    aids_data = sym_df[sym_df['Disease'] == 'AIDS']
    if not aids_data.empty:
        aids_symptoms = set(aids_data['symptoms'].iloc[0])
        test_symptoms = {'high_fever', 'cough'}
        
        overlap = aids_symptoms.intersection(test_symptoms)
        print(f"\nAIDS symptoms: {aids_symptoms}")
        print(f"Test symptoms: {test_symptoms}")
        print(f"Overlap: {overlap}")
        print(f"Overlap ratio: {len(overlap) / len(aids_symptoms):.2%}")
        
        # Find similar diseases
        print("\n🔍 Other diseases with similar symptom overlap:")
        similarities = []
        for _, row in sym_df.iterrows():
            disease = row['Disease']
            disease_symptoms = set(row['symptoms'])
            overlap = disease_symptoms.intersection(test_symptoms)
            overlap_ratio = len(overlap) / len(disease_symptoms) if len(disease_symptoms) > 0 else 0
            
            if len(overlap) > 0:
                similarities.append((disease, overlap_ratio, len(overlap), disease_symptoms))
        
        # Sort by overlap ratio
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 diseases by symptom overlap with test case:")
        for disease, ratio, overlap_count, symptoms in similarities[:10]:
            print(f"   - {disease}: {ratio:.1%} overlap ({overlap_count}/{len(symptoms)} symptoms)")
    
    # Save summary
    summary = {
        "total_diseases": len(disease_counts),
        "total_samples": len(sym_df),
        "total_unique_symptoms": len(symptom_counts),
        "avg_symptoms_per_disease": float(avg_symptoms_per_disease),
        "min_samples_per_disease": int(disease_counts.min()),
        "max_samples_per_disease": int(disease_counts.max()),
        "diseases_with_1_sample": len(single_sample),
        "top_diseases": disease_counts.head(10).to_dict()
    }
    
    with open(os.path.join(BASE, "training_data_analysis.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete! Summary saved to training_data_analysis.json")
    print("=" * 70)


if __name__ == "__main__":
    analyze_training_data()
