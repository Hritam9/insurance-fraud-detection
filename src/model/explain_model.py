import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed/model_features.csv")
X = df.drop(columns=["is_fraud"])

model = joblib.load("models/fraud_model.pkl")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

Path("docs").mkdir(parents=True, exist_ok=True)
plt.title("SHAP Feature Importance")
shap.summary_plot(shap_values[1], X, show=False)
plt.savefig("docs/shap_summary.png", bbox_inches="tight")
plt.close()
print("✅ SHAP summary plot saved to docs/shap_summary.png")
