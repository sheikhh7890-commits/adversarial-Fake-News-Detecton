# Refactored Experiment Scripts

This keeps the existing LIAR + DeBERTa + TextAttack workflow, but moves the duplicated notebook code into reusable modules and script entrypoints.

Original notebooks are now organized under `notebooks/` by project phase:

- `notebooks/01_baseline/`
- `notebooks/02_adversarial_generation/`
- `notebooks/03_adversarial_training/`
- `notebooks/04_evaluation/`

Scripts:

- `python scripts/train_baseline.py`
- `python scripts/generate_adversarial_dataset.py`
- `python scripts/train_adversarial.py`
- `python scripts/evaluate_attack_success.py`
- `python scripts/evaluate_attack_success_multi.py`
- `python scripts/evaluate_clean_accuracy_multi.py`

Shared code lives in `src/liar_adv/`:

- `common.py` for labels, text construction, dataset loading, seeding, and file helpers
- `training.py` for weighted-loss trainer and optimizer/scheduler setup
- `attacks.py` for BERTAttack configuration and checkpoint helpers
- `reporting.py` for clean-evaluation plots and per-class reports

The default arguments mirror the notebook defaults so existing paths and outputs stay aligned unless you override them on the command line.
