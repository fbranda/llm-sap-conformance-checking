"""
STEP 3 — Applicability Filtering (executed for real against the actual
simulated variable table produced in Step 1/1b).
"""
import json
import pandas as pd

df = pd.read_csv("point/itt_dataset.csv")
rules = json.load(open("point/rules_extracted.json"))

available_columns = set(df.columns)

filtering_results = []
for r in rules:
    if r["id"] in ("R1", "R2", "R4", "R6"):
        applicable = {"time_to_event_or_censor_days", "primary_event_observed",
                      "treatment_arm"}.issubset(available_columns)
        reason = "Required variables (time-to-event, event indicator, arm) present in extracted table."
    elif r["id"] == "R3":
        applicable = False
        reason = "No secondary/tertiary outcome variable was extracted in this pilot dataset (out of scope for this run)."
    elif r["id"] == "R5":
        applicable = {"time_to_event_or_censor_days", "primary_event_observed",
                      "treatment_arm"}.issubset(available_columns)
        reason = "Conditional power computable from interim event data."
    elif r["id"] == "R7":
        applicable = {"other_sae", "treatment_arm"}.issubset(available_columns)
        reason = "Other-SAE indicator present by treatment arm."
    elif r["id"] == "R8":
        applicable = {"age_ge65", "primary_event_observed", "treatment_arm",
                      "time_to_event_or_censor_days"}.issubset(available_columns)
        reason = "Age subgroup variable present."
    else:
        applicable = False
        reason = "Unhandled rule id."
    filtering_results.append({**r, "applicable": applicable, "filter_reason": reason})

json.dump(filtering_results, open("point/rules_filtered.json", "w"), indent=2)

n_total = len(filtering_results)
n_applicable = sum(r["applicable"] for r in filtering_results)
print(f"Total rules extracted: {n_total}")
print(f"Applicable rules (Step 3 output): {n_applicable}")
for r in filtering_results:
    print(f"  {r['id']}: applicable={r['applicable']} — {r['filter_reason']}")
