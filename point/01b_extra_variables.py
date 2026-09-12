"""
Extends the Step-1 simulated dataset with the additional variables needed
to exercise rules R7 (safety/Fisher's exact) and R8 (subgroup analysis),
again calibrated to values stated in the public SAP (Section 13.2 expected
rates; Section 12.1.8 subgroup list).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260910 + 1)
df = pd.read_csv("point/itt_dataset.csv")

# Age subgroup (SAP 12.1.8): age >=65 vs <65. Realistic TIA/minor-stroke age mix.
df["age"] = rng.normal(65, 12, size=len(df)).round(1).clip(18, 95)
df["age_ge65"] = (df["age"] >= 65).astype(int)

# Other SAE (non-primary-outcome, non-hemorrhage) — SAP 13.2 "other SAEs"
# Simulate a modest imbalance for illustration (real trial did not report a
# significant difference in most safety endpoints for POINT).
p_sae_placebo = 0.045
p_sae_clop = 0.052
df["other_sae"] = np.where(
    df["treatment_arm"] == "placebo",
    rng.random(len(df)) < p_sae_placebo,
    rng.random(len(df)) < p_sae_clop,
).astype(int)

df.to_csv("point/itt_dataset.csv", index=False)
print(df[["age", "age_ge65", "other_sae"]].describe())
print("SAE counts by arm:")
print(df.groupby("treatment_arm")["other_sae"].agg(["sum", "count"]))
