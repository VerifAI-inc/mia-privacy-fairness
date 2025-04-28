# Artifacts Repository

This repository contains all the artifacts and codebases for two major projects:

- **LiRA** (Membership Inference Attack experiments)
- **OQTA/OTA** (Fairness, Privacy, and Utility trade-offs under advanced attacks)

Both are organized into separate folders with their own structure and dependencies.

---

## Folder Structure

```
Artifacts/
├── LiRA/
│   ├── data_utils.py
│   ├── LiRA_bank_dt.ipynb
│   ├── LiRA_bank_nn.ipynb
│   ├── LiRA_bank_rf.ipynb
│   ├── requirements.txt
│   └── data/
│       ├── bank_dup.csv
│       ├── compas_preprocessed_final.csv
│       └── law_preprocessed.csv
│
└── OQTA_OTA/
    ├── active_poison.py
    ├── cpp_fairness_notebook_bank_dt.ipynb
    ├── data_utils.py
    ├── explainer.py
    ├── fairness_notebook_bank_dt.ipynb
    ├── fairness_notebook_compas_gender_dp_rf_depth6_eps1.ipynb
    ├── fairness_notebook_compas_race_nn.ipynb
    ├── fairness_notebook_law_gender_rf_depth7.ipynb
    ├── fair_ml.py
    ├── membership_infer_attack.py
    ├── metrics_utils.py
    ├── MIA_Attack_Result.py
    ├── mitigators.py
    ├── models.py
    ├── oversample.py
    ├── plot_utils.py
    ├── requirements.txt
    ├── test_algorithms.py
    │
    ├── data/
    │   ├── bank_dup.csv
    │   ├── compas_preprocessed_final.csv
    │   ├── law_preprocessed.csv
    │   └── student-por.csv
    │
    ├── dataset_explorations/
    │   ├── data_exploration_bank_compas.ipynb
    │   ├── data_preprocessing_law.ipynb
    │   └── data/
    │       ├── bank_dup.csv
    │       ├── bank_reduced.csv
    │       ├── bank_reduced_dup.csv
    │       ├── compas-scores-two-years.csv
    │       ├── compas_preprocessed.csv
    │       ├── law_preprocessed.csv
    │       └── lsac.csv
    │
    └── privacy_meter/
        ├── audit.py
        ├── audit_report.py
        ├── constants.py
        ├── dataset.py
        ├── hypothesis_test.py
        ├── information_source.py
        ├── information_source_signal.py
        ├── metric.py
        ├── metric_result.py
        ├── model.py
        ├── utils.py
        └── __init__.py
```

---

## Descriptions

### LiRA

- **Purpose:**
  - Runs Membership Inference Attacks (LiRA) on decision trees, random forests, and neural networks.

- **Contents:**
  - `LiRA_bank_dt.ipynb`, `LiRA_bank_rf.ipynb`, `LiRA_bank_nn.ipynb` — Main experiments.
  - `data_utils.py` — Data loading and preprocessing.
  - `requirements.txt` — Environment setup.

- **Datasets:**
  - `bank_dup.csv`, `compas_preprocessed_final.csv`, `law_preprocessed.csv`

### OQTA_OTA

- **Purpose:**
  - Analyzes Fairness–Privacy–Utility trade-offs under poisoning and membership inference attacks.

- **Contents:**
  - Core scripts: `active_poison.py`, `fair_ml.py`, `membership_infer_attack.py`, `mitigators.py`, `models.py`
  - Helper utilities: `data_utils.py`, `metrics_utils.py`, `plot_utils.py`, `oversample.py`
  - Analysis Notebooks: `fairness_notebook_*` and `cpp_fairness_notebook_*`
  - Attack Results: `MIA_Attack_Result.py`
  - Testing: `test_algorithms.py`
  - `requirements.txt` — Environment setup.

- **Datasets:**
  - `bank_dup.csv`, `compas_preprocessed_final.csv`, `law_preprocessed.csv`, `student-por.csv`

- **Subfolders:**
  - `dataset_explorations/`
    - Notebooks and extra datasets for deeper exploratory data analysis.
  - `privacy_meter/`
    - Privacy evaluation utilities and metrics for attack audits.

---

## Setup Instructions

Each major folder has its own `requirements.txt`. To set up environments separately:

```bash
# For LiRA
cd Artifacts/LiRA
pip install -r requirements.txt

# For OQTA_OTA
cd Artifacts/OQTA_OTA
pip install -r requirements.txt
```

It is recommended to use virtual environments (like `venv` or `conda`) to avoid dependency conflicts.

---

## Notes

- Python 3.8+ is recommended.
- Jupyter Notebooks (`.ipynb` files) should be run inside the corresponding folder.
- Some experiments might take significant computational resources.

---

## Contact

For any questions or clarifications, please reach out to the maintainers of this repository. (inovruzova16235@ada.edu.az, kmammadov14045@ada.edu.az, nhasanova15562@ada.edu.az, bgurbanli14129@ada.edu.az). 

---

*(Last updated: April 2025)*
