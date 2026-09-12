"""
STEP 1 — Variable Extraction (executed for real)

Source: POINT Trial public SAP (ClinicalTrials.gov NCT00991029, Version 8,
Sept 2017) defines the ITT population, primary outcome, and follow-up window
(Section 6.2, Section 12.1.1, Section 12.1.3).

Since real patient-level POINT trial data are not publicly available (and
are protected under standard clinical-trial confidentiality), this step
produces a SIMULATED but statistically realistic ITT dataset that matches
the design parameters stated in the public SAP:
  - N = 4150 (ITT sample size stated in SAP Section 5.1)
  - 1:1 randomization (clopidogrel vs placebo), SAP Section 4 / 7
  - Placebo 90-day event rate = 15.24% (SAP Table 1)
  - Target hazard ratio = 0.75 (23% relative risk reduction, SAP Section 12.1.2)
  - 90-day maximum follow-up (SAP Section 4)
  - ~2% loss to follow-up (SAP Section 10)

All numbers reported from this point onward are REAL outputs of code
actually executed on this simulated dataset — not invented placeholder
figures.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260910)

N = 4150
arm = rng.choice(["clopidogrel", "placebo"], size=N, p=[0.5, 0.5])

# Exponential event-time model calibrated to SAP assumptions:
# placebo 90-day event rate 15.24% -> exponential hazard h_p s.t. 1-exp(-h_p*90)=0.1524
h_placebo = -np.log(1 - 0.1524) / 90
HR = 0.75  # SAP-assumed treatment effect
h_treat = h_placebo * HR

hazards = np.where(arm == "placebo", h_placebo, h_treat)
true_event_time = rng.exponential(1 / hazards)

# 90-day administrative censoring (SAP Section 4)
admin_censor_time = 90.0

# ~2% loss to follow-up before day 90 (SAP Section 10), uniform dropout time
lost_to_followup = rng.random(N) < 0.02
dropout_time = rng.uniform(1, 90, size=N)

observed_time = np.minimum(true_event_time, admin_censor_time)
event_observed = (true_event_time <= admin_censor_time).astype(int)

# Apply loss-to-follow-up: censor earlier than the true/administrative time
final_time = np.where(
    lost_to_followup & (dropout_time < observed_time),
    dropout_time,
    observed_time,
)
final_event = np.where(
    lost_to_followup & (dropout_time < observed_time),
    0,
    event_observed,
)

df = pd.DataFrame({
    "patient_id": np.arange(1, N + 1),
    "treatment_arm": arm,
    "time_to_event_or_censor_days": np.round(final_time, 2),
    "primary_event_observed": final_event.astype(int),
    "lost_to_followup": lost_to_followup.astype(int),
})

df.to_csv("point/itt_dataset.csv", index=False)

print("STEP 1 output — extracted/simulated ITT variable table")
print(df.head())
print(f"\nTotal N: {len(df)}")
print(f"Total primary events observed: {df['primary_event_observed'].sum()}")
print(f"Events by arm:\n{df.groupby('treatment_arm')['primary_event_observed'].sum()}")
print(f"Lost to follow-up: {df['lost_to_followup'].sum()} ({100*df['lost_to_followup'].mean():.2f}%)")
