# SoftwareX policy and Silhouette audit (2026-11-17)

**Status:** exploratory validation on `main`, **NOT** part of the tagged
`v1.0.0` release and **NOT** yet an approved replacement for the paper's
tables. All three jobs completed successfully in
[GitHub Actions run 44039050405](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/44039050405),
using public Online Retail II, reduced DAMADICS and UCI Electricity,
five seeds 42–46, operation-count reward, the original AE budget and temporal
splits. The archived workflow artifacts contain the raw `evaluation.csv`,
`summary.json`, `audit_report.json` and baselines for each dataset.

## Changes evaluated

1. The greedy Q-action resolver uses a *seeded random tie-break* rather than
   choosing the smallest action index. Rows now contain `q_state`,
   `q_unseen_state`, `q_action_visited`, `q_action_visit_count`,
   `q_tie_count`, `q_tie_break`, `q_margin` and a map of valid Q values.
   A trained state and a trained **state–action pair** are distinguished.
2. `test_silhouette_valid` is NaN for a degenerate single-cluster result.
   The old `test_silhouette` remains a **penalized score** (-1 in invalid
   cases) for continuity of reward semantics; it must NOT be reported as a
   standard mean Silhouette. `test_degenerate` and the valid count are
   recorded separately. DBI undefined values remain NaN.
3. `q_learning_fixed` is a matched-configuration comparison constrained to
   K=3, d=8; the standard Q-learning policy can choose K in {3,5} and latent
   dimension in {8,12}. Both may select reuse/update/retrain.

## New Q-learning results (five seeds, held-out evaluation)

| Dataset | Policy | Mean *valid* Silhouette | Degenerate / total | Full fits | Updates | Reuses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Retail II | Always retrain | 0.2821 | 0/60 | 60 | 0 | 0 |
| Retail II | Q-learning | 0.2648 | 0/60 | 32 | 6 | 22 |
| Retail II | Q-learning, fixed K/d | 0.2455 | 0/60 | 11 | 29 | 20 |
| Retail II | Q-learning + gate | 0.2569 | 0/60 | 27 | 22 | 11 |
| DAMADICS | Always retrain | 0.7095 | 0/20 | 20 | 0 | 0 |
| DAMADICS | Q-learning | 0.5997 | 0/20 | 16 | 4 | 0 |
| DAMADICS | Q-learning, fixed K/d | 0.5526 | 0/20 | 8 | 12 | 0 |
| Electricity | Always retrain | 0.3779 | 1/60 | 60 | 0 | 0 |
| Electricity | Q-learning | 0.3382 | 1/60 | 40 | 4 | 16 |
| Electricity | Q-learning, fixed K/d | 0.3576 | 0/60 | 30 | 17 | 13 |
| Electricity | Q-learning + gate | 0.3115 | 1/60 | 44 | 15 | 1 |

The true comparison of quality and efficiency must consider **valid scores,
degeneracy counts, configuration changes and training actions jointly**.

### Was reuse learned or caused by a tie?

| Dataset | Policy | Total reuses | Selected Q action trained | Selected Q action **not** trained | Reuses due to Q tie |
| --- | --- | ---: | ---: | ---: | ---: |
| Retail II | Q-learning | 22 | 20 | 2 | 2 |
| Retail II | Q-learning, fixed K/d | 20 | 16 | 4 | 4 |
| Retail II | Q-learning + gate | 11 | 10 | 1 | 1 |
| DAMADICS | Q-learning | 0 | 0 | 0 | 0 |
| Electricity | Q-learning | 16 | 11 | 5 | 5 |
| Electricity | Q-learning, fixed K/d | 13 | 0 | 13 | 13 |
| Electricity | Q-learning + gate | 1 | 0 | 1 | 1 |

**Critical:** A Q-action being trained does not, by itself, establish that it
was optimal or consistently learned. More importantly, the 13/13 fixed-K/d
Electricity reuses were selected at tied, **untrained** Q-action values.
They must NOT be described as learned reuse. To test a trained-only
decision rule, an explicit action-coverage or fallback policy and a separate
five-seed ablation are still necessary.

### Degenerate clustering example

Reduced DAMADICS `reuse_when_available`: 14 of 20 results have invalid
(single-cluster) Silhouette. The penalized mean was -0.6073, but the mean
among the **six valid** results is 0.3089. The former must be labeled as a
penalized quality score, not mean Silhouette. The 14/20 failure rate belongs
beside the valid mean in a revised table.

## Submission decision

**Do not submit the existing manuscript as if its Table 3, abstract percentages,
or reuse mechanism reflect these audited runs.** Retain published tag v1.0.0
unchanged. Before a new final release, consider an explicit trained-action
fallback/coverage policy for unseen states, then rerun all comparisons and
issue a fresh tagged version with the final CSV files.


## Follow-up: trained-action-only coverage ablation

**Independent verification completed:** [GitHub Actions run 44039942375](https://github.com/NTDANH-VLU/AdaptiveClusterLab-SoftwareX/actions/runs/44039942375), all three dataset jobs passed. The tested commit is `e60c71aa68494d6dc1e7d618464485197ac682d8`. Its artifacts contain full five-seed CSV files, the unchanged K-Means comparisons, and `audit_report.json`. No raw private data were committed.

Two additional exploratory policies:

- `q_learning_covered`: retains K in {3,5}, d in {8,12}, but permits an evaluation action only when that exact **state–action pair** had a Q update during policy development. If none of the valid actions was trained, it falls back to `(K=3,d=8,retrain)`, logging `q_coverage_fallback=True`.
- `q_learning_fixed_covered`: applies the same coverage rule within fixed K=3,d=8, to separate action adaptation from configuration selection.

| Dataset | Policy | Valid Silhouette | Degenerate / total | Full AE fits | Updates | Reuses | Reuses with Q training | Coverage fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retail II | Always retrain | 0.2821 | 0/60 | 60 | 0 | 0 | n/a | 0 |
| Retail II | Q-learning, covered | 0.2678 | 0/60 | 33 | 7 | 20 | 20/20 | 5 |
| Retail II | Fixed K/d, covered | 0.2554 | 0/60 | 16 | 27 | 17 | 17/17 | 6 |
| DAMADICS | Always retrain | 0.7095 | 0/20 | 20 | 0 | 0 | n/a | 0 |
| DAMADICS | Q-learning, covered | 0.5997 | 0/20 | 16 | 4 | 0 | n/a | 0 |
| DAMADICS | Fixed K/d, covered | 0.5526 | 0/20 | 8 | 12 | 0 | n/a | 0 |
| Electricity | Always retrain | 0.3779 | 1/60 | 60 | 0 | 0 | n/a | 0 |
| Electricity | Q-learning, covered | 0.3684 | 1/60 | 49 | 0 | 11 | 11/11 | 28 |
| Electricity | Fixed K/d, covered | 0.3728 | 0/60 | 54 | 6 | 0 | n/a | 33 |

The covered action policy eliminates **untrained Q reuse** by construction,
while exposing how often the model must fall back to retraining. Its results
remain below always retraining in mean valid Silhouette on the three datasets.
The percentages of full fits avoided relative to always retrain are 45%
(Retail II), 20% (DAMADICS), and about 18.3% (Electricity); these describe
full-AE-fit counts, **not measured total runtime savings**.

The fixed-K/d covered ablation is particularly instructive: on Electricity,
it gives **zero** covered reuses and 33 fallbacks, where the previous
unrestricted fixed-K/d policy produced 13 reuses from untrained Q actions.
Thus the earlier result should not be interpreted as evidence of learned reuse.

**Conclusion:** the three originally raised validity issues are now auditable,
and a conservative action-coverage alternative has been evaluated. The new
results must be reviewed and incorporated into the manuscript before
submission. The public `v1.0.0` tag is intentionally unchanged; these changes
currently live on `main` and need a fresh versioned release if adopted.
