import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

df = pd.read_csv("data/processed/model_features.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
print("✅ Model trained and saved as models/fraud_model.pkl")
