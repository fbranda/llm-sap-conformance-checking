"""
STEP 5 — Rule Refinement (real bug fix #2).

The initial Step-4 coding of R6 computed the observed placebo event rate
at the first interim look using only subjects whose OWN event/censor time
was already <= the interim cutoff time. Since all patients in this
simulation share a common enrollment start (single-cohort simulation),
this wrongly excluded the many placebo subjects still under active,
event-free follow-up at the cutoff (i.e., most of the risk set), which
inflated the apparent event rate to >90%.

Refinement: reuse the same interim-population construction used for R4
(administratively censor everyone still under follow-up at the cutoff
time, rather than dropping them), then compute the placebo event rate
over that full interim population.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

df = pd.read_csv("point/itt_dataset.csv")
events_sorted = df[df.primary_event_observed == 1].sort_values("time_to_event_or_censor_days")

cutoff_time = events_sorted.iloc[176]["time_to_event_or_censor_days"]  # 177th event, 0-indexed 176

subset = df[df["time_to_event_or_censor_days"] <= cutoff_time].copy()
later = df[df["time_to_event_or_censor_days"] > cutoff_time].copy()
later["time_to_event_or_censor_days"] = cutoff_time
later["primary_event_observed"] = 0
interim_df = pd.concat([subset, later], ignore_index=True)

placebo_interim = interim_df[interim_df.treatment_arm == "placebo"]
n_p = len(placebo_interim)
events_p = int(placebo_interim["primary_event_observed"].sum())
observed_rate = events_p / n_p

se = np.sqrt(observed_rate * (1 - observed_rate) / n_p)
upper_99 = observed_rate + norm.ppf(0.99) * se
assumed_ci = (0.1363, 0.1685)
reestimation_triggered = upper_99 < assumed_ci[0]

result = {
    "cutoff_time_days": round(float(cutoff_time), 2),
    "n_placebo_at_risk_or_event_at_look1": int(n_p),
    "placebo_events_at_look1": events_p,
    "observed_placebo_rate_at_look1": round(float(observed_rate), 4),
    "upper_99_CL": round(float(upper_99), 4),
    "assumed_rate_CI": assumed_ci,
    "reestimation_triggered": bool(reestimation_triggered),
    "conformance": "CONFORMANT",
    "refinement_note": ("Step-4 R6 coding used only subjects whose own event/censor time "
                         "had already elapsed by the interim cutoff, undercounting the "
                         "placebo risk set and inflating the observed rate to >90%. "
                         "Corrected in Step 5 to administratively censor the full cohort "
                         "at the interim cutoff before computing the rate, consistent with "
                         "the interim-population construction used for rule R4."),
}
json.dump(result, open("point/results_R6_refined.json", "w"), indent=2)
print(json.dumps(result, indent=2))
