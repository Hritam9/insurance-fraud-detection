#!/bin/bash
set -e
python src/data_prep/generate_synthetic_data.py
python src/data_prep/clean_data.py
python src/data_prep/feature_engineering.py
python src/model/train_model.py
python src/model/evaluate_model.py
python src/model/explain_model.py
