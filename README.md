# Adversarial Fake News Detection on the LIAR Dataset

This repository contains a notebook-first NLP research project that studies adversarial robustness for fake-news / claim-veracity classification on the LIAR dataset. The core model is `microsoft/deberta-v3-large`, trained as a 6-class classifier and evaluated under BERTAttack.

The project started as a set of Jupyter notebooks for experimentation and was later refactored into reusable Python scripts and shared modules without changing the core training or evaluation logic.

## Project Goal

The main objective is to compare:

- a baseline DeBERTa model trained on the clean LIAR dataset
- adversarially generated examples created with BERTAttack
- adversarially trained models that mix clean and attacked samples
- clean accuracy versus adversarial robustness across multiple runs

## Dataset

- Dataset: `ucsbnlp/liar`
- Task: 6-class classification
- Labels:
  - `false`
  - `half-true`
  - `mostly-true`
  - `true`
  - `barely-true`
  - `pants-fire`

Input text is built from LIAR fields using the pattern:

```text
statement [SEP] SUBJECT: ... [SEP] CONTEXT: ...
```

## Repository Structure

```text
.
├── docs/
│   ├── PROJECT_TIMELINE.md
│   └── README_refactor.md
├── models/
│   └── final_model/
├── notebooks/
│   ├── 01_baseline/
│   │   └── deberta_baseline_training.ipynb
│   ├── 02_adversarial_generation/
│   │   └── adversarial_dataset_generation.ipynb
│   ├── 03_adversarial_training/
│   │   └── adversarial_training_fixed.ipynb
│   └── 04_evaluation/
│       ├── attack_success_rate_evaluation.ipynb
│       ├── attack_success_rate_multi_model.ipynb
│       └── clean_accuracy_multi_model.ipynb
├── results/
│   └── ...
├── scripts/
│   ├── train_baseline.py
│   ├── generate_adversarial_dataset.py
│   ├── train_adversarial.py
│   ├── evaluate_attack_success.py
│   ├── evaluate_attack_success_multi.py
│   └── evaluate_clean_accuracy_multi.py
├── src/
│   └── liar_adv/
│       ├── common.py
│       ├── training.py
│       ├── attacks.py
│       └── reporting.py
├── requirements.txt
└── README.md
```

## Workflow

### Phase 1: Notebook-based experimentation

The original workflow lives in the notebooks:

- [deberta_baseline_training.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\01_baseline\deberta_baseline_training.ipynb)
- [adversarial_dataset_generation.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\02_adversarial_generation\adversarial_dataset_generation.ipynb)
- [adversarial_training_fixed.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\03_adversarial_training\adversarial_training_fixed.ipynb)
- [attack_success_rate_evaluation.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\04_evaluation\attack_success_rate_evaluation.ipynb)
- [attack_success_rate_multi_model.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\04_evaluation\attack_success_rate_multi_model.ipynb)
- [clean_accuracy_multi_model.ipynb](C:\Users\sheik\OneDrive\Desktop\New folder\notebooks\04_evaluation\clean_accuracy_multi_model.ipynb)

These notebooks were used to:

- train the baseline DeBERTa model
- generate adversarial examples with TextAttack / BERTAttack
- build non-overlapping clean + adversarial training sets
- retrain robust models
- compare clean accuracy and attack success rates

### Phase 2: Script refactor

The later refactor moves shared logic into:

- [common.py](C:\Users\sheik\OneDrive\Desktop\New folder\src\liar_adv\common.py)
- [training.py](C:\Users\sheik\OneDrive\Desktop\New folder\src\liar_adv\training.py)
- [attacks.py](C:\Users\sheik\OneDrive\Desktop\New folder\src\liar_adv\attacks.py)
- [reporting.py](C:\Users\sheik\OneDrive\Desktop\New folder\src\liar_adv\reporting.py)

Runnable script entrypoints are:

- [scripts/train_baseline.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\train_baseline.py)
- [scripts/generate_adversarial_dataset.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\generate_adversarial_dataset.py)
- [scripts/train_adversarial.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\train_adversarial.py)
- [scripts/evaluate_attack_success.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\evaluate_attack_success.py)
- [scripts/evaluate_attack_success_multi.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\evaluate_attack_success_multi.py)
- [scripts/evaluate_clean_accuracy_multi.py](C:\Users\sheik\OneDrive\Desktop\New folder\scripts\evaluate_clean_accuracy_multi.py)

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Typical Usage

Train baseline model:

```powershell
python scripts/train_baseline.py
```

Generate adversarial training data:

```powershell
python scripts/generate_adversarial_dataset.py
```

Train adversarial model:

```powershell
python scripts/train_adversarial.py
```

Evaluate attack success rate across saved models:

```powershell
python scripts/evaluate_attack_success_multi.py
```

Evaluate clean accuracy across saved models:

```powershell
python scripts/evaluate_clean_accuracy_multi.py
```

## Notes

- The notebooks are kept as the original research workflow.
- The scripts are a cleanup layer for reproducibility and code reuse.
- Large trained models and result folders are ignored in Git by default through `.gitignore`.
- If you want to track selected result files on GitHub, you can remove `results/` from `.gitignore` or commit a smaller `reports/` subset manually.

## Documentation

For the project history and development phases, see [PROJECT_TIMELINE.md](C:\Users\sheik\OneDrive\Desktop\New folder\docs\PROJECT_TIMELINE.md).
