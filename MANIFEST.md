# SpikeRPGNN — Final Submission Package (MANIFEST)

**Manuscript:** *Commanded Firing Rates in Spiking Graph Neural Networks: A Capacity-Matched Evaluation Discipline*
**Author:** Kehui Jiang (Hunan Women's University, Changsha, China)
**Target journal:** *Neural Networks* (Elsevier), section *Learning Systems*
**Package snapshot:** frozen **2026-11-16** (mathematical-audit pass: Poincare origin/general map convention mismatch corrected (origin-form expmap lost a spurious 1/2; general logmap gained the missing 1/lambda_x factor); membrane reset equation aligned to the implemented hard reset; threshold clamp corrected to the one-sided form actually coded; GDC gated fused distance now described in Sec. 4.4; highlights hook replaced with the verdict-migration result; review line numbers enabled and pinned by new gate G5, 46 -> 47 venue-compliance checks).
**Main PDF:** 44 pages, 0 LaTeX errors, 0 undefined citations

---

## 1. File inventory

| Path | Contents |
|---|---|
| `manuscript/` | Main LaTeX source (`SpikeRPGNN.tex`), `references.bib`, Elsevier CAS class/style (`cas-dc.cls`, `cas-common.sty`, `model1-num-names.bst`), and `tables/` (14 generated table fragments, incl. `energy_silicon_table.tex`) |
| `manuscript/SpikeRPGNN.pdf` | **Compiled main manuscript (44 pp)** |
| `supplementary/` | `Supplementary.tex` + compiled `Supplementary.pdf` (9 pp, full proofs of Proposition 4.1 / Theorem 5.1) |
| `cover_letter/` | `Cover_Letter.tex` + compiled `Cover_Letter.pdf` (4 pp) |
| `figures/` | 6 vector figures (PDF) + PNG previews: `graphical_abstract`, `chart_ablation`, `chart_energy`, `chart_pareto`, `fig_c_sensitivity`, `energy_silicon_projection` |
| `code/` | `spikergnn/` (model package), `msg_baseline/` (capacity-matched MSG* baseline), `scripts/` (reproduction + verification scripts, incl. `neuromorphic_silicon_projection.py` and `make_silicon_figure_pdf.py`), `figureCode/` (figure-generation scripts), `requirements.txt`, `mutag_sop_checkpoint.pt` |
| `results/` | 36 result JSONs — every number in the manuscript is derived from these (incl. `energy_silicon_projection.json`, `gat_baselines_v30.json`, `sigma0_control.json` and the six `s0_*` shards it is merged from, `dd_credit_assignment_signif.json` behind Table `tab:dd_signif`, and `analysis_v30.json`, whose `significance` block is the 60-cell BH-FDR family) |
| `verification/` | 9-gate machine-verification suite (`check_all.sh` + 16 `*.py`; 9 of them are gating gates, the rest are helpers/specs) |
| `README.md` | How to reproduce / run the gates |

## 2. Machine-verification gates (re-run 2026-10-22, gnn_env: torch 2.11.0+cu126)

`bash verification/check_all.sh` → **all 9 gates present in tree PASSED** (run with `PY=<project interpreter>`; the default `python3` is not the project environment):

| # | Gate | Result |
|---|---|---|
| 1 | Data-consistency audit (result JSONs) | **836** checks pass |
| 2 | Table-number traceability (manuscript ↔ data) | **694** checks pass |
| 3 | Hyperparameter disclosure (runtime reflection vs declaration) | **28** match (RTX 2080 Ti / CUDA) |
| 4 | Algorithm-to-code mapping | **28** steps match |
| 5 | Euclidean-control completeness | **10 / 10** |
| 6 | Table bold = best-in-column | **82** cells |
| 7 | Narrative-claim recomputation (significance tests + σ=0 control) | **63** checks pass |
| 8 | Venue compliance (*Neural Networks* / Elsevier mandatory items) | **47** checks pass |
| 9 | Cross-file consistency (manuscript / tables / cover letter / MANIFEST / README) | **63** checks pass |

Note: gates 3–6 read `code/spikergnn` and require `torch`+`torch_geometric`. On this machine they were run under a workspace venv with the same `gnn_env` interpreter (torch 2.11.0+cu126) plus `torch_geometric`.

## 3. Key reported numbers (all traced to `results/`)

- **Energy (analytic SOP model, 0.9 pJ/SOP vs 4.6 pJ/FLOP, T=8-aware):** SpikeRPGNN **8.88 µJ/graph** vs capacity- & backbone-matched dense Proto-GIN 13.61 µJ/graph, i.e. a **34.8 %** reduction. (A single-step count that omits the T=8 temporal integration factor would report 1.11 µJ / 91.8 %; that figure is superseded and is retained only as this cautionary note.)
- **Silicon projection (constant-rescaling, NOT a device measurement) — sign reversed under the T-aware accounting.** The same independently re-derived spike-driven SOP counts are re-priced onto published per-event constants of real fabricated chips — Intel Loihi **23.6 pJ/SOP** (Davies et al., 2018) and IBM TrueNorth **26 pJ/SOP** (Merolla et al., 2014). Because the spiking model is charged for all **T=8** integration steps, on the *same* substrate it now **overruns** the dense Proto-GIN reference by roughly **218–233 %** (SpikeRPGNN 3.33×, Euclidean control 3.18×), replacing the superseded 58–60 % saving that a single-step count had produced. **This reversal is reported in the manuscript rather than suppressed:** the paper no longer claims a substrate-matched efficiency gain. What the controllable r̄ provides on this axis is a selectable, reproducible operating point, not a guaranteed saving, since cost scales as T·r̄.
- **Rate controllability:** 33 / 35 runs inside target band [0.3, 0.5]; 1-shot std ≤ 0.005 on PROTEINS/ENZYMES.
- **Four adverse cells stated up front (BH-FDR over the complete 60-cell family), after the n>=20 power extension:** Proto-GIN Rdt-M5k 1-shot -4.39 pp (q<0.001, paired n=20); Proto-GAT DD 1-shot -6.70 pp (q=0.032, Welch) and PROTEINS 5-shot -3.35 pp (q=0.010, Welch); and the Euclidean spiking control PROTEINS 5-shot -4.25 pp (q=0.028, paired n=20; the accuracy side of a significant Pareto domination). The legacy Proto-GIN DD 1-shot deficit dissolved with the extension (-2.5 pp, q=0.062), and the apparent Rdt-M5k 1-shot Euclidean deficit dissolved as well (-1.5 pp, q=0.126); only the five Proto-GAT rows remain unequal-count (20 vs 5, Welch). Against the four adverse cells, SpikeRPGNN wins **23** cells significantly, against five of the six baselines in the family (MSG*, Proto-GCN, Proto-SAGE, Proto-GIN, Proto-GAT) and none against the Euclidean spiking control.
- **Capacity-matched MSG\*:** fails as a graph-level few-shot learner under the unified protocol (chance on DD/ENZYMES, near-chance on MUTAG) — no accuracy claim rests on it.
- **Seven-variant ablation + seven-point λ_sr sweep** + BH-FDR testing over **all 60** capacity-matched cells (paired *t*-tests where the two arms share a run count; Welch's *t*-test on the 12 cells where they do not). Effect sizes are Cohen's *d*<sub>z</sub> on the paired cells and standard pooled Cohen's *d* on the Welch cells; the table note states which is which.
- **DD credit-assignment study (§6.5), recomputed and reported as found.** The stored *q*-values had been produced without the BH step-up monotonicisation, which inflated every *q*. Recomputing with the standard step-up procedure (cross-checked against `scipy.stats.false_discovery_control`) drops the minimum *q* from 0.0586 to 0.0195: **4 of the 15** comparisons are now significant at *q*<0.05, and **none of the four involves the estimator the paper adopts**. The §6.5 prose was rewritten to say so instead of repeating the superseded "no scheme attains a significant advantage" claim.

## 4. Fixes applied in this final packaging pass (2026-09-30)

1. **Cross-file q-value consistency (cover letter ← manuscript/JSON).** Cover letter previously carried stale Proto-GIN q-values; corrected to match the manuscript and the authoritative `results/analysis_v30.json`: DD 1-shot `0.0227→0.021`, DD 5-shot `0.7924→0.770`, Rdt-M5k 5-shot `0.0810→0.068`. (Rdt-M5k 1-shot `0.0123` was already correct.)
2. **Gate-count consistency (cover letter & README).** Stale "788 / 543" updated to the actual **796 / 546**.
3. **Dangling citation removed.** Energy table caption cited `\citep{Pfeiffer18}`, but `references.bib` had no such entry → rendered as `[?]` in the PDF. Aligned the table with the main text, which attributes the 0.9 pJ/SOP constant to `\citep{Han16}`; the table now reads `\citep{Han16}`. (Prior reference audits had mis-flagged `Pfeiffer18`/SLAYER as fabricated — it is a real paper, arXiv:1805.01221; the resolution here is to cite the same source the body already uses, keeping the package internally consistent. **Recommend the author double-check the 0.9 pJ/SOP provenance** — Han16 vs a dedicated neuromorphic-energy source — before final upload.)
4. **All three PDFs recompiled** (manuscript 28 pp / supplementary 9 pp / cover letter 3 pp), 0 errors, 0 undefined citations.

## 4b. Addendum — Loihi/TrueNorth silicon validation (2026-09-30)

1. **New manuscript subsection "Validation Against Published Neuromorphic Silicon"** (§sec:silicon) added after the Energy-Consumption-Analysis section. It re-prices the already-measured spike-driven SOP counts onto published per-event constants of real neuromorphic silicon (Loihi 23.6 pJ/SOP, TrueNorth 26 pJ/SOP) as a *same-operation-count, constant-rescaling projection* — explicitly **not** a measurement on physical hardware (no Loihi/TrueNorth boards exist in this environment).
2. **New table + figure:** `manuscript/tables/energy_silicon_table.tex` (Table `tab:energy_silicon`) and `figures/energy_silicon_projection.pdf` (Figure `fig:silicon`, dependency-free vector PDF). Both generated by `code/scripts/neuromorphic_silicon_projection.py` + `code/scripts/make_silicon_figure_pdf.py`; source data in `results/energy_silicon_projection.json`.
3. **Two new BibTeX entries** (`Davies18`, `Merolla14`) added to `references.bib`; cited in §sec:silicon and the revised Energy-model *Note*. Both constants web-verified against the primary papers (Davies et al., IEEE Micro 38(1):82–99, 2018; Merolla et al., Science 345(6197):668–673, 2014 — the latter states 26 pJ per synaptic event).
4. **Honest negative-result framing retained:** the subsection states up front that Loihi/TrueNorth per-op energy exceeds a 45-nm MAC, so the *absolute* edge vs a von-Neumann accelerator is not claimed; under the T-aware accounting the *substrate-matched* (same-hardware) result is an overrun of roughly **218–233 %**, not a saving.
5. **Gate ② extended:** `verification/verify_numbers.py` now parses `energy_silicon_table.tex` and binds every cell to `energy_silicon_projection.json`, cross-checks the constants against `score_roadmap_numbers.json`, and asserts the §sec:silicon prose anchors (`\label{sec:silicon}`, `\input`, figure include, the 23.6/26 pJ/SOP constants, and the same-substrate projection sign). Gate ② rose **546 → 574** checks; all 6 gates remain green.
6. **Manuscript 29 pages at the time of this addendum**; the current page count is recorded in the header above.

## 4c. Pre-submission consistency pass (2026-10-21)

Driven by the pre-submission consistency audit. All fixes are text/recomputation-level; no experiment was re-run.

1. **Proto-GAT significance signs.** The 8 Proto-GAT entries in `results/analysis_v30.json` had been inserted with the reversed difference convention (mean(baseline) − mean(SpikeRPGNN)); all 8 mean_diff / t / cohen_d values were recomputed from the raw per-run accuracies under the sanctioned convention (two-sided p and BH q-values are sign-invariant and were verified unchanged). Table 6 and the Sec 6.1 narrative were regenerated: significant baseline wins are **2** (both Proto-GIN 1-shot) and SpikeRPGNN wins **15** cells significantly (previously stated as 5 / 12).
2. **Gate blind spots closed.** `verify_narrative_claims.py` printed mean_diff mismatches without judging them — the check now gates, together with a sign-consistency check; `verify_numbers.py` no longer hard-codes the literal word "five" for the adverse-cell count. Both were negative-tested (deliberately corrupted data → exit 1).
3. **Numbered citations.** natbib switched to `[numbers,sort&compress]`; bibtex re-run under `model1-num-names.bst`; the PDF was probed and renders `[n]` citations with zero author-year residue.
4. **Abstract ≤ 250 words.** 300 → 247 words under the strict counter (math content included); every gate-checked phrase preserved.
5. **T-aware ablation energies.** `tab:spike_rate_ablation` energy column 0.51/1.35 µJ (T=1 residue) → 4.07/10.80 µJ, with the three prose restatements; the derived "≈96 %" figure was recomputed as 77–85 %.
6. **σ=0 narrative aligned.** Implementation-details text, the contribution list, and cover-letter contribution 5 restated to match the measured control: the membrane noise supplies *reproducibility*, not the control authority (at σ=0 the regularizer still moves the rate on PROTEINS p=0.0013 and ENZYMES p=0.0002; MUTAG locks onto period-2 plateaus).
7. **AI-declaration consistency.** Cover letter and README now carry the same "generative AI for English language polishing only" declaration as the manuscript.
8. **Package hygiene.** Orphan `orphaned build files` removed from the submission manuscript directory; stale 2026-10-16 manuscript snapshots archived out of the working tree; `gat_baselines_v30.json` and `code/figureCode/` added to the bundle; MANIFEST/README/gate counts refreshed; the post-tree root README (still targeting Neurocomputing) replaced. *Correction:* an earlier archive shipped 13 LaTeX build artefacts (`.aux/.log/.bbl/.blg/.out/.abs` under `cover_letter/`, `manuscript/`, `supplementary/`) even though this list already claimed the package was clean — the archive is rebuilt with those extensions excluded, and the version stamp above now matches the artefact name. The working tree still holds them because they are regenerated by every compile; only the delivered archive is filtered.

## 4d. Audit-fix pass (2026-10-22)

All changes are text/recomputation-level; no experiment was re-run except the DD credit-assignment recomputation, which reuses the stored per-run data.

1. **Pareto → rate-constrained calibration.** The manuscript's framing of `L_sr = λ_sr(r̄−r*)²` as a *weighted-sum scalarization of accuracy against energy* was false: the second term carries no accuracy information, so the objective is a **penalty relaxation of the constraint r̄ = r***, not a scalarization of a bi-objective problem. The claim was downgraded throughout (title, short title, 2 highlights, keywords, abstract, contributions 2/3/4, Related Work gap paragraph, §4.4 title and equations, Sec 5 proposition scope, Sec 6 wording, figure/table captions, discussion, conclusion — 37 anchors in the main manuscript, plus the cover letter, which had not been touched by the first pass). The *empirical* Pareto relation between measured accuracy–energy pairs is retained, and is now explicitly labelled as a description of measurements rather than of the objective.
2. **DD credit-assignment BH-FDR recomputed.** See §3. Minimum *q* 0.0586 → 0.0195; 0 → 4 significant; §6.5 rewritten.
3. **Significance family completed 58 → 60.** Proto-GAT's two 10-vs-5 cells (DD 1-shot, Rdt-M5k 1-shot) had been excluded from the family even though the method is reported as capacity-matched in the main table; excluding them shrank the correction family and made every surviving *q* smaller, i.e. biased in the direction that flatters the proposed method. They are now included via Welch's test. On the 12 n-mismatched cells the effect size is reported as the standard pooled Cohen's *d* rather than *d*<sub>z</sub>, which is undefined without pairing; the table note states the distinction. Consequence: wins 15 → 16, and no previously significant cell changed verdict (max *q* shift 0.0152).
4. **Factual corrections.** Energy ordering prose (the spiking entries are the lowest *pair* among spiking entries, not the lowest entries overall); SOP formula given its missing `T×` factor; contribution anchors repointed; the λ=0.1/0.5 exceptions restated as three with dataset attribution; test-type sentence corrected to "paired where the run counts match, Welch where they do not"; the abstract's "geometric-consistency bound" removed because Theorem 5.1's spectral-norm condition `‖W‖₂ ≤ 1−ε` is violated by every trained model (15/15 spectral norms > 1, up to 2.206).
5. **Gate corrections.** `verify_numbers.py` no longer expects 58 rows (Proto-GAT added to the Welch-covered suites; the message now names "untestable cells" rather than "unpaired"); `verify_crossfile_consistency.py` publishes the venue-compliance count; `verify_narrative_claims.py`'s stale Welch effect-size claim (2.98) updated to the pooled value 2.64 that the manuscript now quotes.
6. **Three direct predecessors added to Related Work.** §2.3 previously cited only generic activity-penalty work and skipped the three lines closest to this paper. Added, each verified against its publisher record before the BibTeX entry was written: `\citet{Yan22}` (BPSR: an $L_2$ firing-rate penalty inside backpropagation, *Frontiers in Neuroscience* 16:760298, DOI 10.3389/fnins.2022.760298), `\citet{Meem26}` (energy-aware spike budgeting that explicitly calls spike activity a first-class control signal and reports modality-dependent accuracy--spike-activity behaviour, *Neuromorphic Computing and Engineering*, DOI 10.1088/2634-4386/ae8627), and `\citet{Na22}` (AutoSNN, ICML 2022, arXiv:2201.12738 — NAS under a fitness that multiplies accuracy by a normalized spike-count penalty). The paragraph now states the difference in formulation explicitly: those works penalize activity or search for low-activity architectures, so the realized rate stays an outcome; this paper imposes $\bar{r}=r^{*}$ as an equality constraint with a commanded target. The bibliography stands at **45** entries, all cited. The `references.bib` header caption (which still named *Neurocomputing*) was corrected to *Neural Networks*.
7. **Terminology unified.** The measured quantity is now called the **firing rate** everywhere; "spike rate" survives only inside the proper name of the regularizer. The lone "firing frequency" instance was removed. 14 anchors rewritten by `code/scripts/a one-off terminology script`, which aborts if any anchor is not matched exactly once.
8. **New gate: the separately uploaded highlights file is bound to the manuscript.** Elsevier wants highlights as a distinct editable file, so the package carries `submission/highlights.txt` *and* the manuscript carries a `highlights` environment — two sources of truth for one artefact. The uploaded file had been left at the old Pareto wording while the manuscript moved on. `verify_crossfile_consistency.py` now parses both and fails on any count or text mismatch (and re-checks the ≤ 85-character rule). Cross-file gate rose 43 → 55 checks.
9. **Readability.** The Introduction now states what the paper contributes within the first ~185 words (previously the reader reached it only after 500+ words of background); the 1,420-word results paragraph was split at its (iii) block; the Figure 4 caption was cut from 262 to ~175 words with every gate-checked number preserved.

10. **The Discussion now states the cell where our own Euclidean ablation is strictly better.** The geometry controls were previously summarised only as "neutral / not resolved". That understates them: on **PROTEINS 5-shot** the Euclidean spiking control is more accurate by 5.73 pp ($63.73\%$ vs $58.00\%$; $p=0.018$, $q=0.057$, $|d|=1.72$) **and** cheaper under the analytic SOP model ($10.95$ vs $11.40$ µJ/graph at effectively identical measured rates, $\bar r = 0.392$ vs $0.393$), i.e. it **Pareto-dominates** SpikeRPGNN there. Because the two rates match, the energy gap is the cost of the Riemannian operators themselves, not a rate-control failure. On Rdt-M5k 1-shot the control is also ahead by 4.0 pp, but the energy model is not evaluated on that benchmark so domination cannot be assessed. A new Discussion paragraph reports this, notes that the largest effect size in the whole comparison still misses significance after correction (so the cell is "under-powered and adverse", not "null"), and records that it rules out any claim that the Riemannian component pays for itself on those two cells.

11. **DvM attribution — no change to this manuscript.** The review flagged a possible conflict between this paper's statement that the DvM estimator originates with Sun24/MSG and a related manuscript by the same author. The author confirmed the attribution: DvM is due to Sun24/MSG, so *this* manuscript is correct as written. Any correction belongs to the separate DvM manuscript (`NEUNET-D-26-06671`), which is outside this package.

## 5. Limitations (stated in manuscript, cover letter, and README)

1. Energy figures are **analytic estimates, not hardware measurements**. The Loihi/TrueNorth column is a *constant-rescaling projection* of the measured SOP counts onto published per-event constants — it is not a measurement on physical neuromorphic silicon. The ±50 % per-constant sensitivity analysis bounds the analytic projection; priced at Loihi/TrueNorth per-event constants the same-substrate comparison reverses sign (a ~218–233 % overrun), while the absolute edge vs a von-Neumann accelerator is explicitly **not** claimed (24–26 pJ > 4.6 pJ/MAC).
2. The method holds the best mean accuracy in **0 of 10** dataset–shot cells; every cell's best mean belongs to a baseline, by 0.4–6.2 pp. The training objective carries a firing-rate constraint that the accuracy-only baselines do not carry, so an accuracy-only ranking does not test the paper's claim; the Pareto relation quoted in the paper describes the *measured* accuracy–energy pairs, and is **not** the training objective (the rate penalty carries no accuracy term, so it is a constraint, not a weighted scalarization).
3. **Empirical scale is modest**: 5 benchmarks, 5 runs/config (10 on DD & Rdt-M5k 1-shot), fixed CPU budget.

## 6. ⚠️ Journal-name flag (action required before upload)

**Journal decision resolved (2026-10-15).** This package previously targeted *Neurocomputing* while the project record named *Neural Networks*. The target is now **Neural Networks (Elsevier), section *Learning Systems***: the cover-letter addressee, salutation, scope paragraph, title, gate counts and page count have all been updated to match. The `cas-dc` document class is retained — Elsevier's official CAS class, accepted by both journals, so it is not a journal-specific artifact and needed no change. A further consistency pass on 2026-10-21 (§4c) aligned the cover letter, MANIFEST and README with the corrected numbers; no journal-target change was involved.

## 7. Reproduction

```bash
# Regenerate analysis + tables, then verify every number
python code/scripts/analyze_results_v30.py
python code/scripts/paper_tables_to_tex.py
PY=<interpreter-with-torch+torch_geometric> bash verification/check_all.sh

# Independent SOP-count check (3 separate implementations agree exactly)
python code/scripts/verify_sop_counting.py --checkpoint code/mutag_sop_checkpoint.pt
```

Experiments ran on CPU (PyTorch 2.13.0+cpu); no GPU/neuromorphic hardware was used for the reported results.
