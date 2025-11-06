import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/cleaned_claims.csv")

region_map = {"Low": 0, "Medium": 1, "High": 2}
claim_map = {"Vehicle": 0, "Health": 1, "Home": 2}

df["region_risk_score"] = df["region_risk"].map(region_map)
df["claim_type_code"] = df["claim_type"].map(claim_map)

features = df[["claim_amount", "region_risk_score", "claim_type_code", "customer_age", "past_claims", "is_fraud"]]
features.to_csv("data/processed/model_features.csv", index=False)
print("✅ Feature engineered dataset created at data/processed/model_features.csv")
