# Participant

Kadi Sadaraka

# Disclosure

- Agent product and version: Claude Code 2.1.319 (iterations 1 to 6), 2.1.322 (iterations 7 to 12, and iterations 13 to 20 in a third session).
- LLM(s): Claude Opus 5.5 (`claude-opus-5-5`) in all three sessions.
- ML model families used: Random Forest (iters 1-6, 8, 12, 13, 14), LightGBM (7, 15, 17, 19, 20), SVM (9), Ensemble of Random Forest + SVM (10, 16), MLP (11), Ensemble of Random Forest + LightGBM (18). **The final model is LightGBM.**
- Compute (CPU/GPU, RAM): CPU only, AMD Ryzen 3 3250U, 10 GB RAM. Each iteration took 7 to 65 seconds.
- Human interventions (iteration, what you did, why):
  - Before iteration 1: the participant wrote Part 2 of `program.md` (strategy, decision rules, roadmap, stop rule).
  - After iteration 6: the first session stopped at 6 of 20 iterations under the original stop rule (4 consecutive runs with no improvement above 0.01), with only Random Forest tried.
    - The participant rewrote the stop rule so the run could not stop before (a) a gradient-boosting model, (b) an SVM or MLP and (c) an ensemble of two families had been tried.
    - The participant raised the iteration limit from 20 to 25, updated the roadmap, and restarted the agent with the same one-line instruction.
    - This overrode the agent's decision to stop.
  - After iteration 12: the second session had stopped at 12 under the revised rule, with iteration 8 declared final.
    - The participant reopened the run and rewrote the Part 2 roadmap and stop rule: noise checks with two extra seeds first (iters 13-14), then LightGBM on the iter 8 features, a weighted RF+SVM vote, an NM_M hybrid, XGBoost and optional steps.
    - The new stop rule needs 3 consecutive non-improvements after iteration 14, measured against 0.01 and the seed spread.
    - The participant restarted the agent with "Read program.md and follow it." This again overrode the agent's decision to stop.
  - No code, notes or results were edited by a human. The participant only approved commands, including the agent's `pip install` of the pinned `lightgbm==4.7.0` and `xgboost==3.4.1` before iteration 7.

# Iteration summary

- Total iterations run, and the iteration limit you set: **20 run**, limit 20 for iters 1-6, then 25. No iteration failed or was interrupted.
- The run stopped at 20 under the final Part 2 rule: iters 18, 19 and 20 were 3 consecutive runs with no improvement above 0.01 and the seed spread.
- Noise checks: the iter 8 configuration scored 0.5181, 0.5171 and 0.5221 over three seeds (spread 0.005). The iter 17 configuration scored 0.5365 and 0.5289 over two seeds (spread 0.008). Per-well scores move by up to about 0.03 between seeds.

| # | Hypothesis / change | CV weighted F1 | Kept? | What was learned |
|---|---|---|---|---|
| 1 | Baseline RF, 200 trees, 7 raw logs | 0.4870 | Reference | Errors stay inside the non-marine (F1-F3) and marine (F4-F9) groups. F5 and F1 are the worst classes. |
| 2 | Add window-3 rolling mean/std and lag1/lead1 of the 5 curves, within each well | 0.5014 | Yes | +0.0144, spread over 5 wells. F6/F8 improved, F1/F5 did not. |
| 3 | Z-score the 5 curves within each well, replacing the absolute values | 0.4899 | No | F9 collapsed: WELL_02 is all F9, and z-scoring it erases its high PE. F1 improved (+0.086). |
| 4 | Add centred gradient and second difference | 0.4977 | No | No net effect. |
| 5 | `class_weight="balanced"` | 0.4819 | No | F5 rose, while F2/F6/F8 lost more. F1 is a transfer problem, not a class-prior problem. |
| 6 | Add stratigraphic unit order and LM flag from Formation | 0.4980 | No | F9 +0.029, offset by F6 -0.037. |
| 7 | LightGBM on the iter 2 features | 0.4988 | No | F5/F9 up, F3/F6 down. |
| 8 | Per-well z-scores **added alongside** the absolute values (RF) | 0.5181 | Yes | +0.0167, all 7 wells up. Relative levels absorb the shift between wells, while absolute PE still defines F9. |
| 9 | RBF SVM on standardised iter 8 features | 0.4951 | No | F1 nearly doubled, F9 -0.155. |
| 10 | RF + SVM soft vote, equal weights | 0.5071 | No | The SVM's confident probabilities dominate. F9 -0.082. |
| 11 | MLP (64,32) | 0.4809 | No | Like the SVM, but stronger: F1 up, F9 collapsed. |
| 12 | Iter 8 RF + 3-sample probability smoothing | 0.5119 | No | F2/F3 and F6/F8 errors are whole beds, not isolated samples. |
| 13 | Noise check: iter 8, SEED 7 | 0.5171 | Measurement | The overall score moves 0.001; per-well up to 0.025. |
| 14 | Noise check: iter 8, SEED 123 | 0.5221 | Measurement | RF seed spread 0.005 (< 0.01), so the 0.01 rule is the binding threshold. |
| 15 | LightGBM (iter 7 settings) on the iter 8 features | 0.5270 | No (+0.0089 < 0.01) | F1 0.280 -> 0.470, F5 0.040 -> 0.164, but F9 0.579 -> 0.434. |
| 16 | RF 0.7 / SVM 0.3 soft vote | 0.5082 | No | Fixes iter 10's F9 loss, but the SVM still damages F2/F3; WELL_05/09 down. |
| - | (Roadmap step 17, SVM only on NM_M=1: not run) | est. 0.5126 | - | Estimated from the archived iter 8/9 confusion matrices: the SVM's F2/F3 losses outweigh its F1 gain. |
| 17 | Iter 15, but single-facies training wells (WELL_02) are z-scored with pooled stats of the other training wells | **0.5365** | **Yes (best)** | Self z-scoring put all of WELL_02's F9 rows at z = 0, while real F9 has PE_z of about +1.5. The fix gives F9 +0.113 vs iter 15 and keeps the F1/F5 gains. |
| 18 | RF + LightGBM soft vote, equal weights | 0.5318 | No | The RF's strong F2 vote overrules LightGBM's F1 calls (F1 -0.079). |
| 19 | Noise check: iter 17, SEED 7 | 0.5289 | Measurement | LightGBM seed spread is about 0.008. On seed means, iter 17's config beats iter 8's by about 0.014. |
| 20 | Iter 17 with only well-relative curves (absolute levels dropped) | 0.4871 | No | F1 -0.206, F9 -0.219. Absolute PE/ILD levels carry real signal; relative values only help as an addition. |

# Final model

- Declared final iteration and why this one: **Iteration 17**, CV weighted F1 0.5365 (macro F1 0.4949, accuracy 0.5490; per-well 0.430 WELL_03 to 0.669 WELL_04).
  - It has the best CV score of the 20 runs and passed both thresholds against iter 8 (+0.018 single run).
  - A seed rerun (iter 19, 0.5289) confirmed the gain on seed means: 0.5327 vs 0.5191 for the iter 8 configuration.
  - Its gain over iter 8 comes from three wells (WELL_07 +0.058, WELL_03 +0.038, WELL_04 +0.023). Two wells fell slightly (WELL_01 -0.014, WELL_09 -0.021), both within the measured per-well seed noise.
  - Iter 18 (0.5318) is within the seed spread but uses two models, so the simpler iter 17 is preferred.
  - Being the best of 20 compared runs, its CV score is probably about 0.01 optimistic.
- Features, model, key hyperparameters: 32 features, all computed within each well, sorted by Depth, from that well's rows only:
  - the 7 inputs `GR`, `ILD_log10`, `DeltaPHI`, `PHIND`, `PE`, `NM_M`, `RELPOS`;
  - for each of the 5 curves, a centred window-3 rolling mean and std, the values one sample above and below, and a per-well z-score.
  - **Design choice to disclose (per-well normalisation):** at prediction time, every well, including each test well, is z-scored with the mean and std of its own rows. No labels and no other wells are used, and no model parameter is fitted on test rows, but the transform uses unlabelled statistics of each test well.
  - **Training-time rule:** a training well that contains only one facies (in practice the synthetic WELL_02, all F9) is z-scored with the pooled mean and std of the other training wells, computed inside `fit()`. The rule is generic and not keyed on the well name. It never applies at prediction time.
  - Model: `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)`, lightgbm 4.7.0. No class weights.

# Analysis

- Which facies / wells are still hard, and why:
  - **F5 (mudstone)** is still the worst class (0.167; 110 of 217 called F6). It occurs in thin beds between carbonates and has no distinctive log signature. LightGBM with relative features is the first configuration to get it well above zero without losing the large classes (iter 5's class weights and iter 12's smoothing failed).
  - **F2/F3** (191 and 244 mutual errors) and **F6/F8** (122 and 112) remain the main confusions. Iter 12 showed these are whole-bed errors, and every model family (RF, LightGBM, SVM, MLP) makes the same ones. The 5 curves do not separate these physically similar facies.
  - **WELL_03** (0.430) is the worst well because it holds 61% of all F1 rows, so holding it out removes most F1 training data. F1 rose from 0.280 (iter 8) to 0.494, but 150 of 259 F1 rows are still called F2.
  - **The main finding of the run:** WELL_02 (synthetic, all F9) interacts badly with per-well normalisation.
    - Replacing absolute values with z-scores (iter 3) or adding z-scores to boosting (iter 15) made F9 collapse, because a single-facies well z-scored against itself sits at z = 0.
    - Normalising such a well with pooled training statistics (iter 17) fixed this.
    - Iter 20 then showed that absolute levels are still needed: relative values alone lose F1 and F9.
  - **Smooth models** (SVM, MLP) transfer F1 well, but they lose F9 and F2/F3. In ensembles with the RF (iters 10, 16) or as a non-marine hybrid (estimate), their losses outweighed their gains.
  - **CV-to-test risk:** per-well F1 moves by about 0.03 with the seed alone, and F1 and F9 scores depend on which wells hold those facies. The two test wells could therefore score several points away from 0.5365.
- What you would try next:
  - Average LightGBM over several seeds inside one configuration (bagging) to cut the about 0.008 seed variance.
  - A window-based 1D-CNN over a short depth window, which was not tried.
  - Features aimed at F5/F6, for example the contrast between a sample and the beds above and below it.

# Agent setup

- Changes made to program.md and what worked or did not work with the agent:
  - Part 1 was never changed. Part 2 was rewritten twice by the participant (after iter 6 and after iter 12); see Human interventions.
  - What worked:
    - **One change per iteration with per-well and per-class checks.** Iter 15's per-class result (F1/F5 up, F9 down) pointed to a specific mechanism. A free data check confirmed it (WELL_02's F9 rows have z = 0, real F9 +1.5), and a targeted one-line fix (iter 17) gave the best model.
    - **The noise-check step added after iteration 12.** It gave a measured threshold and showed that per-well swings of 0.02-0.03 are seed noise. The agent extended it on its own with a seed rerun of the new best model (iter 19), which halved the apparent gain but confirmed it.
    - **Estimating a planned step from archived results.** The agent skipped roadmap step 17 with a recorded estimate (0.5126) instead of spending a run.
  - What did not work as well:
    - The fixed roadmap order: the agent deviated where the evidence pointed elsewhere and recorded why in notes.md.
    - Counting a noise check as a non-improvement under the stop rule is a conservative reading. It ended the run at 20 of 25, with XGBoost and the 1D-CNN untested.
