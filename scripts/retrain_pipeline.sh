#!/bin/bash
set -euo pipefail

echo "Starting retrain pipeline..."
python src/data_prep/generate_synthetic_data.py
python src/data_prep/clean_data.py
python src/data_prep/feature_engineering.py
python src/model/train_model.py
python src/model/evaluate_model.py

if [ "${SKIP_SHAP:-true}" = "true" ]; then
  echo "SKIP_SHAP is true — skipping SHAP explainability step"
else
  echo "Running SHAP explainability"
  python src/model/explain_model.py
fi

echo "Pipeline finished"
