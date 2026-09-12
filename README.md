# LLM-Orchestrated Statistical Analysis Conformance Checking

Code and results accompanying the manuscript:

> **Orchestrating Large Language Models to Support Medical Statistical Analysis Conformance Checking**
> [Author names], Unità di Statistica Medica ed Epidemiologia Molecolare, Università Campus Bio-Medico di Roma
> Submitted to the *Journal of Biomedical Informatics*

This repository adapts the orchestrated LLM conformance-checking architecture of
Leonardi et al. (2026), *"Orchestrating large language models to support medical
process conformance checking,"* *Neuroscience Informatics* 6:100294, from clinical
**process** conformance to statistical **analysis-plan (SAP)** conformance.

## What this pipeline does

Given a clinical trial's public Statistical Analysis Plan (SAP) and a dataset:

1. **Variable Extraction** — build a structured variable table from the data source (Stage 1).
2. **Analysis Plan Extraction** — extract atomic `IF... THEN...` statistical rules from the SAP text (Stage 2).
3. **Applicability Filtering** — keep only rules checkable against the available variables (Stage 3).
4. **Rule Coding** — generate executable Python code for each applicable rule (Stage 4).
5. **Rule Refinement** — fix bugs and edge cases in the generated code (Stage 5).
6. **Conformance Checking** — execute the refined rules and compute the **Statistical Analysis
   Conformance Indicator (SACI)** per rule (Stage 6).

See `common/pipeline_diagram.svg` for the full architecture diagram (Fig. 1 in the manuscript).

## Important note on the data used

**No real patient-level data were used anywhere in this repository.** The three trials piloted
here (POINT, SHINE, STAAMP) have confidential, non-public patient-level datasets. For each trial,
Stage 1 instead constructs a **simulated cohort calibrated to the design parameters explicitly
stated in that trial's own public SAP** (sample size, randomization ratio, target event rates,
hazard ratio, follow-up window). This is disclosed at every point in the manuscript and in the
code (see the docstring at the top of each `run_pipeline.py` / `01_variable_extraction.py`).
All statistics reported are the real, actually-computed output of the code in this repository
run against these disclosed simulated cohorts — not placeholder or invented numbers.

## Repository structure

```
.
├── README.md                          <- this file
├── requirements.txt                   <- Python dependencies
├── LICENSE
├── common/
│   ├── pipeline_diagram.svg           <- Fig. 1 (architecture diagram)
│   ├── fig_rules_per_trial.png        <- Fig. 2 (rules extracted/applicable/bugs per trial)
│   ├── fig_point_boundary.png         <- Fig. 3 (POINT O'Brien-Fleming boundary vs. observed Z)
│   └── fig_all_rules_conformance.png  <- Fig. 4 (per-rule conformance overview, all 3 trials)
├── point/                             <- POINT trial pilot (NCT00991029)
│   ├── 01_variable_extraction.py      <- Stage 1: SAP-calibrated simulated ITT cohort
│   ├── 01b_extra_variables.py         <- Stage 1 (extension): age subgroup + SAE variables
│   ├── rules_extracted.json           <- Stage 2 output: 8 atomic rules from the real SAP text
│   ├── 02_applicability_filtering.py  <- Stage 3: applicability check (real dataset vs rules)
│   ├── rules_filtered.json            <- Stage 3 output
│   ├── 03_rule_coding_and_conformance.py  <- Stages 4 & 6: rule code + real executed results
│   ├── 04_refinement_R2_bugfix.py     <- Stage 5: real bug found & fixed (rule R2)
│   ├── 04_refinement_R6_bugfix.py     <- Stage 5: real bug found & fixed (rule R6)
│   ├── results.json                   <- Stage 6 raw output (pre-refinement R2/R6)
│   ├── results_R2_refined.json        <- Stage 6 output after R2 refinement
│   └── results_R6_refined.json        <- Stage 6 output after R6 refinement
├── shine/                             <- SHINE trial pilot (NCT01369069)
│   ├── rules_extracted.json           <- Stage 2 output: 7 atomic rules from the real SAP text
│   ├── run_pipeline.py                <- Stages 1, 3, 4, 6 combined
│   └── results.json                   <- Stage 6 output (real, post-bugfix)
└── staamp/                            <- STAAMP trial pilot (NCT02086500)
    ├── rules_extracted.json           <- Stage 2 output: 6 atomic rules from the real protocol text
    ├── run_pipeline.py                <- Stages 1, 3, 4, 6 combined
    └── results.json                   <- Stage 6 output (real, post-bugfix)
```

## Reproducing the results

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt

# POINT trial pilot
python point/01_variable_extraction.py
python point/01b_extra_variables.py
python point/02_applicability_filtering.py
python point/03_rule_coding_and_conformance.py
python point/04_refinement_R2_bugfix.py
python point/04_refinement_R6_bugfix.py

# SHINE trial pilot
python shine/run_pipeline.py

# STAAMP trial pilot
python staamp/run_pipeline.py
```

Each script prints its Stage output as JSON to stdout and writes it to the corresponding
`results*.json` file. Random seeds are fixed in every script, so results are exactly
reproducible.

## Source SAP documents (public)

| Trial | ClinicalTrials.gov ID | SAP document |
|---|---|---|
| POINT | [NCT00991029](https://clinicaltrials.gov/study/NCT00991029) | SAP Version 8, September 2017 |
| SHINE | [NCT01369069](https://clinicaltrials.gov/study/NCT01369069) | SAP Version 2.0, January 2015 |
| STAAMP | [NCT02086500](https://clinicaltrials.gov/study/NCT02086500) | Protocol Version 2.7, January 2019 |

## Three genuine coding bugs found and fixed

In the spirit of full transparency (and consistent with the manuscript's Sections 4.5 and 4.9),
this pipot's execution surfaced three real implementation bugs, each caught only because the
generated code was actually run against data rather than just reviewed as text:

1. **POINT rule R2** — checked the wrong subject-level flag for "missing data" (a propensity
   flag rather than actual censoring status). Fixed in `point/04_refinement_R2_bugfix.py`.
2. **POINT rule R6** — miscalculated the interim placebo event-rate denominator. Fixed in
   `point/04_refinement_R6_bugfix.py`.
3. **SHINE / STAAMP** — a non-shuffled arm assignment (SHINE) and an incorrectly shaped
   stratified-table array passed to `statsmodels` (STAAMP); both fixed directly in
   `shine/run_pipeline.py` and `staamp/run_pipeline.py`, with the fix documented inline.

## Disclosure on AI-assisted execution

Stages 1–5 of this pipeline were, in this pilot, executed by a single LLM (Claude, Anthropic)
acting in sequential, role-specific prompts, rather than by the multi-vendor LLM configuration
(Gemini 2.5 Flash / NotebookLM / Gemini 3 Pro-Preview) used in the original process-conformance
architecture this work adapts. This is discussed as a methodological point in Section 4.7 of
the manuscript. Stage 6 is fully deterministic Python code (no LLM involved).

## Citation

If you use this code, please cite the manuscript (details to be added upon publication) and
the original architecture:

```bibtex
@article{leonardi2026orchestrating,
  title={Orchestrating large language models to support medical process conformance checking},
  author={Leonardi, Giorgio and Montani, Stefania and Striani, Manuel and Canessa, Alessandro and Ferrandi, Delfina},
  journal={Neuroscience Informatics},
  volume={6},
  pages={100294},
  year={2026},
  publisher={Elsevier}
}
```

## License

MIT License — see `LICENSE`.
