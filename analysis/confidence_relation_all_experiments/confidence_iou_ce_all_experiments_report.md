# Confidence vs IoU / Center Error: All Experiments Audit

## Coverage
- prediction files found: 1194
- experiments included: 284
- matched rows: 34547
- scored rows: 33615
- unscored rows: 932

## Overall (All Scored Rows)
- IoU Pearson r: 0.24450795875580963 (perm p=0.000999000999000999)
- IoU Spearman rho: 0.37557732223720564 (perm p=0.000999000999000999)
- CE Pearson r: -0.20271559643020592 (perm p=0.000999000999000999)
- CE Spearman rho: -0.4142869083893078 (perm p=0.000999000999000999)
- IoU delta high20-low20: 0.18187347438552603 (p=0.0004997501249375312)
- CE delta low20-high20: 14.440748641702948 (p=0.0004997501249375312)

## LOSO vs Non-LOSO
- LOSO n_scored: 33615
- LOSO IoU Pearson: 0.24450795875580963 | CE Pearson: -0.20271559643020592
- Non-LOSO n_scored: 0
- Non-LOSO IoU Pearson: None | CE Pearson: None

## Experiment-Level Distribution
- experiments with valid correlations: 284
- median Pearson(score,IoU): 0.14074277670718716
- median Pearson(score,CE): -0.20645411026769472
- support ratio (Pearson IoU>0 and CE<0): 0.8767605633802817

## Artifacts
- all rows CSV: c:/Users/User/Desktop/code/Traking/analysis/confidence_relation_all_experiments/confidence_iou_ce_all_rows.csv
- experiment stats CSV: c:/Users/User/Desktop/code/Traking/analysis/confidence_relation_all_experiments/experiment_level_stats.csv
- summary JSON: c:/Users/User/Desktop/code/Traking/analysis/confidence_relation_all_experiments/confidence_iou_ce_all_experiments_summary.json++