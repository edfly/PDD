# SpikeRPGNN — Submission Package for *Neural Networks* (Elsevier), section *Learning Systems*

**Manuscript:** *SpikeRPGNN: Controllable Spike Rates through Rate-Calibrated Spiking
Graph Neural Networks on Riemannian Product Manifolds*
**Author:** Kehui Jiang (Hunan Women's University, Changsha, China)
**Submitted to:** *Neural Networks* (Elsevier), section *Learning Systems*

---

## 1. Package contents

| Directory | Contents |
|---|---|
| `manuscript/` | Main manuscript (LaTeX source, compiled PDF, bibliography, generated table fragments) |
| `supplementary/` | Supplementary material (complete proofs of Proposition 4.1 and Theorem 5.1, additional experimental detail) |
| `cover_letter/` | Cover letter (LaTeX source and PDF) |
| `code/` | Implementation: `spikergnn/` package, `msg_baseline/` dependency, reproduction scripts, the trained MUTAG model checkpoint used for the operation-count verification, and `figureCode/` (figure-generation scripts) |
| `results/` | Result files (JSON) from which every number in the manuscript is derived |
| `figures/` | Figures included in the manuscript |
| `verification/` | Automated verification suite that re-derives the manuscript's numbers from `results/` |

## 2. Reproducing the reported numbers

Every number appearing in the manuscript is regenerated from the files in
`results/` by the scripts in `code/scripts/`.

**Data snapshot note (per-run arrays).** The `accs` arrays in
`results/analysis_v30.json` for the five 5-shot decision settings were rewritten
when those cells were extended from 5 to 20 runs per arm. Their first five
entries are therefore *not* the original initial-review snapshot, and the
published 2x2 factorial table (Table 4), whose four corners were fixed at five
paired seeds, must be read against that earlier snapshot. In particular,
recomputing the six factorial interactions directly from the current `accs[:5]`
reproduces all three unextended 1-shot cells (e.g. ENZYMES 1-shot
$t(4)=7.36$, $p=0.0018$) but **not** the 5-shot cells, because the Euclidean
control's first five entries were overwritten by extension runs. The published
Table 4 numbers were produced by `code/scripts/recompute_factorial_paired.py`
from the pre-extension snapshot, which is retained as
`results/analysis_v30_pre_extension.json` (5-run arrays; e.g. the Euclidean
control's PROTEINS 5-shot mean is 63.73 there). When checking Table 4, use
that snapshot or the published values rather than slicing the current arrays; when
checking the main accuracy and significance tables, use the current
`analysis_v30.json`. This is a data-versioning caveat, not a discrepancy in
the reported results.

```bash
# Regenerate the analysis and the manuscript tables
python code/scripts/analyze_results_v30.py
python code/scripts/paper_tables_to_tex.py

# Verify that every number in the manuscript matches its source file.
# Point PY at the project interpreter: the gates need numpy/torch/torch_geometric,
# which the system default python3 may not have.
PY=/path/to/project/python bash verification/check_all.sh
```

`bash verification/check_all.sh` runs **nine gates in this bundle**
(data consistency; table-number traceability; hyperparameter disclosure;
algorithm-to-code mapping; Euclidean-control completeness; table
bold-marking; narrative-claim recomputation; *Neural Networks* venue
compliance; and cross-file consistency across manuscript, tables, cover
letter, MANIFEST and README). Exit status 0 means every gate *listed for
this tree* was present and passed, and the closing banner states how many
ran and names any that did not.

One further gate -- the watchdog run-cost audit (`exp_timing_audit.py`) -- and the
two informational reports (`exp_scorecard.py`, `exp_plan_audit.py`) live in the
**full authoring tree**, because they read artefacts a submission bundle does
not ship (the per-run wall-clock logs under `spikergnn_results/`, written while
experiments execute). They are listed under `NOT RUN IN THIS TREE` and are
never counted as passing. Conversely, a gate listed in `check_all.sh` that is
missing from this bundle fails the run rather than being skipped: a suite that
can silently lose a gate prints a green light it did not earn.

Expected output when the package is intact:

| Check | Result |
|---|---|
| Data consistency | 836 checks pass |
| Table-number traceability | 694 checks pass |
| Hyperparameter disclosure | all match (28 reflections) |
| Algorithm-to-code mapping | 29 steps match |
| Euclidean-control completeness | 10 / 10 |
| Table bold-marking | 82 cells |
| Narrative-claim recomputation | 63 checks pass |
| Venue compliance (*Neural Networks* / Elsevier) | 47 checks pass |
| Cross-file consistency (5 documents) | 63 checks pass |

## 3. Operation-count verification

The manuscript reports energy under an *analytic* synaptic-operation (SOP)
accounting model: per-operation constants are literature values, not
measurements of a specific device. To confirm that the operation-count side of
this accounting is nevertheless exact, the spike-driven SOP count was
independently re-derived by traversing each emitted spike over its incident
edges using three separate implementations that share no code path:

```bash
# A: production-path hook; B: standalone PyTorch reference; C: pure-NumPy reference
python code/scripts/verify_sop_counting.py                       # random-init model
python code/scripts/verify_sop_counting.py \
       --checkpoint code/mutag_sop_checkpoint.pt                 # trained MUTAG model
```

All three reconstructions agree exactly. The trained MUTAG model
(`code/mutag_sop_checkpoint.pt`) used for this check was produced by
`code/scripts/train_mutag_sop_checkpoint.py` under the same protocol as the
main experiments; its measured firing rate is $\bar{r}=0.419$ against the
$r^{*}=0.4$ target.

## 4. Environment

Experiments were run on CPU under PyTorch 2.13.0+cpu with NumPy 2.5.1 and
SciPy 1.18.0. Python dependencies are listed in `code/requirements.txt`.
No GPU or neuromorphic hardware was used.

The verification suite needs `numpy`, `torch` and `torch_geometric` (all in
`code/requirements.txt`) because `verify_hyperparams.py` reflects the declared
hyperparameters off a *constructed* model and the number/narrative gates
recompute statistics with NumPy. If they are not available, the affected check
fails and the run exits non-zero -- an unchecked claim is not a passing one.
Always select the interpreter that carries these dependencies explicitly;
the default `python3` on a clean machine is often a different one:

```bash
PY=/path/to/project/python bash verification/check_all.sh
# If the packages live outside that interpreter's own environment instead:
EXTRA_PYTHONPATH=/path/containing/torch_geometric bash verification/check_all.sh
```

## 5. Declaration of limitations

Three limitations are stated in the manuscript and in the cover letter, and are
repeated here so that they travel with the package:

1. **Energy figures are analytic estimates, not hardware measurements.** No
   neuromorphic silicon was used; energies charge per-architecture operation
   counts at literature constants scaled by measured firing rates. A ±50%
   sensitivity analysis bounds the projection.
2. **The method does not hold the best mean accuracy in any of the 10
   dataset–shot cells.** The best mean in every cell belongs to a baseline
   (including the hyperbolic baseline HGCN on MUTAG 1-shot) by 0.4–6.2 percentage
   points. This is reported as the expected signature of a method whose
   objective carries a firing-rate constraint that the accuracy-only
   baselines do not carry. The Pareto relation quoted in the paper describes
   the *measured* accuracy--energy pairs; the training objective itself is a
   penalty relaxation of the constraint $\bar{r}=r^{*}$, not a weighted
   scalarization of accuracy against energy (the penalty carries no accuracy
   term).
3. **Empirical scale is modest.** Five graph-classification benchmarks, five
   runs per configuration (ten for DD and Rdt-M5k 1-shot), fixed CPU budget.

## 6. Ethics and authorship

This work is original, unpublished, and not under consideration elsewhere.
The author declares no competing interests. No human subjects, animal
experiments, or personally identifiable data were involved. The research
was supported by the Hunan Provincial Natural Science Foundation of China
(Grant No. 2026JJ81141); the funder had no role in study design, data
collection and analysis, decision to publish, or preparation of the
manuscript.

## 7. Figure provenance and AI image-tool statement

All figures in the manuscript — including the graphical abstract — were produced
entirely from source scripts under `figureCode/` (Python/Matplotlib and
hand-written TikZ/PDF). **No generative-AI image tools were involved** at any
stage of figure creation. The figure-to-script mapping is:

| Figure | Source script under `figureCode/` |
|---|---|
| Fig. 1 (Frèchet mean) | `fig_frechet_mean.py` |
| Fig. 2 (architecture) | `fig_architecture.py` |
| Fig. 3 (spiking neuron) | `fig_spiking_neuron.py` |
| Fig. 4 (Pareto) | `generate_charts.py` |
| Fig. 5 (energy) | `generate_charts.py` |
| Fig. 6 (silicon projection) | `make_silicon_figure_pdf.py` |
| Fig. 7 (ablation) | `generate_charts.py` |
| Fig. 8 (curvature sensitivity) | `generate_charts.py` |
| Graphical abstract | `generate_charts.py` (`plot_graphical_abstract`) |

Every script is vector-first (PDF) at 300 DPI with a Times New Roman / IEEE
style sheet. All figures are generated by these scripts; no generative-AI image
tool was involved in producing any figure. This is consistent with the
manuscript's *Declaration of generative AI and AI-assisted technologies in
the writing process*, which records that generative AI tools assisted only
with English language polishing.
