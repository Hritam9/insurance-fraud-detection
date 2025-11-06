import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

df = pd.read_csv("data/processed/model_features.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

model = joblib.load("models/fraud_model.pkl")
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

print("✅ Model Evaluation Results")
print(f"Accuracy: {accuracy_score(y, pred):.3f}")
print(f"Precision: {precision_score(y, pred):.3f}")
print(f"Recall: {recall_score(y, pred):.3f}")
print(f"AUC: {roc_auc_score(y, prob):.3f}")
