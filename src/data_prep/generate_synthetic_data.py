import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
Path("data/raw").mkdir(parents=True, exist_ok=True)

def generate_data(n=1000):
    claim_ids = [f"C{i:05d}" for i in range(1, n+1)]
    claim_amount = np.random.randint(2000, 50000, n)
    region_risk = np.random.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2])
    claim_type = np.random.choice(["Vehicle", "Health", "Home"], n)
    customer_age = np.random.randint(18, 70, n)
    past_claims = np.random.randint(0, 5, n)
    is_fraud = np.random.choice([0, 1], n, p=[0.85, 0.15])

    df = pd.DataFrame({
        "claim_id": claim_ids,
        "claim_amount": claim_amount,
        "region_risk": region_risk,
        "claim_type": claim_type,
        "customer_age": customer_age,
        "past_claims": past_claims,
        "is_fraud": is_fraud
    })
    df.to_csv("data/raw/insurance_claims.csv", index=False)
    print("✅ Synthetic data generated at data/raw/insurance_claims.csv")

if __name__ == "__main__":
    generate_data()
