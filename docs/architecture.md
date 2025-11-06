# Architecture Overview

### Flow
1. **Data Generation** → Synthetic claim data created.
2. **ETL** → Clean and transform.
3. **ML Model** → RandomForest classifier trained.
4. **Explainability** → SHAP insights.
5. **Deployment** → Flask API + Streamlit dashboard.
6. **CI/CD** → GitHub Actions retrains daily.

### Components
- `/src/data_prep` — Data engineering
- `/src/model` — Model pipeline
- `/dashboard` — Visualization
- `/src/api` — REST endpoint
