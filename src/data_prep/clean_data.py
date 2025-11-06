import pandas as pd
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
raw = pd.read_csv("data/raw/insurance_claims.csv")

# Fill missing and normalize
raw["claim_amount"] = raw["claim_amount"].fillna(raw["claim_amount"].median())
raw["customer_age"] = raw["customer_age"].clip(18, 70)

raw.to_csv("data/processed/cleaned_claims.csv", index=False)
print("✅ Cleaned data saved to data/processed/cleaned_claims.csv")
