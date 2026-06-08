# NMF Analysis of Multiple Sclerosis Phenotypes

This repository contains Python scripts used to identify latent symptom modules in patients with Multiple Sclerosis using Non-negative Matrix Factorization (NMF).

## Project Overview

Clinical phenotype data were aggregated at the patient level and represented as 17 symptom domains. NMF models with k = 2–5 components were evaluated to identify stable symptom modules.

The selected four-module solution identified:

1. Sensory–Visual–Pain
2. Ataxic–Spastic–Falls
3. Cognitive–Psychologic–Fatigue
4. Autonomic–Bladder–Bowel

## Repository Structure

* `data/` – patient-level phenotype matrices
* `scripts/` – analysis scripts
* `results/` – generated figures and output tables

## Requirements

```bash
pip install -r requirements.txt
```

## Example

```bash
python scripts/nmf_heatmaps.py \
    --input data/ms_patient_summary.csv \
    --output-dir results/heatmaps
```

## Author

Daniel B. Hier, MD
Missouri University of Science and Technology
