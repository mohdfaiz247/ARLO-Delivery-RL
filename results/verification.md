# Execution verification

Executed locally on 9 November 2026 using Python 3.12.14.

- `python -m pytest -q`: **74 passed**, 8.95 seconds.
- `python scripts/train.py --algorithm q_learning --episodes 15000 --seed 42`: completed and saved model/history; final 250-episode moving return −56.66.
- `python scripts/train.py --algorithm sarsa --episodes 15000 --seed 42`: completed and saved model/history; final 250-episode moving return −56.49.
- `python scripts/run_experiments.py`: completed with 15,000 episodes per agent, 1,000 held-out orders, three-scenario robustness and independent balanced diagnostics. CSVs, model snapshots and six standalone HTML charts exported successfully. The default experiment was executed twice; the displayed mean costs agreed.
- Streamlit started at `http://127.0.0.1:8501` and rendered in the in-app browser. Visually inspected the overview and live simulator. Executed the default automatic episode through delivery: Q-learning chose combined intervention at approval, final delay was two days, total cost R$82 versus R$95 for matched no intervention. This is one illustrative random outcome, not an average-performance estimate.
- Streamlit AppTest loaded all eight sections and exercised training, evaluation, manual next-stage/full-run, automatic full-run, risk-based order input and simulation reset.
- Windows Application Control blocked the native Arrow library during initial tests. Tables were implemented using escaped HTML; the final application and tests do not require native Arrow table rendering. No system security settings were changed.

The full default evaluation and scenario results are in the accompanying CSV files. Numerical dependency versions are recorded in `experiment_metadata.json`; direct application package versions are pinned in `../requirements-tested.txt`.
