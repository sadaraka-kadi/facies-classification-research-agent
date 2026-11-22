# Research program: Well Log Facies Classification

You are an autonomous ML research agent. Work in this folder and follow this program.
Read `README.md` first for the data description.

This file has two parts. **Part 1 is fixed by the organizers. Part 2 belongs to the participant.**

<!-- RULES:BEGIN -->
## Part 1 – Competition rules (LOCKED: do not edit, move or delete this part)

`evaluate.py` refuses to run if anything between the `RULES:BEGIN` and `RULES:END` markers
has been changed. These rules override anything written in Part 2.

**Goal.** Maximise the **CV weighted F1** reported by `python evaluate.py run`, in a way that
will also hold on two unseen wells.

**Iterations.** The competition sets no maximum number of iterations. Every
`python evaluate.py run` is one iteration and is logged, including failed or interrupted runs;
the baseline run is iteration 1. `python evaluate.py check` and `status` are free. The
participant sets an iteration limit in Part 2, and `evaluate.py` will not start a run beyond it.
Only the 10 iterations with the best CV score (plus the final one) are submitted in full, but
the log and the notes must cover every iteration.

**Boundaries.**

- Edit only `train.py` (and `requirements.txt` to uncomment or add pinned packages). Keep the
  `fit(train_df)` / `predict(model, df)` interface and keep all modelling code in `train.py`.
- Never edit `evaluate.py`, `data/`, or anything already written by `evaluate.py` in `results/`
  (`log.csv`, `final.json`, `iter_NN/`). Never delete or reset `results/`.
- Allowed models: Random Forest, XGBoost, LightGBM, SVM, MLP, 1D-CNN, LSTM, and ensembles of them.
- One iteration = one fixed configuration. No grid/random/Bayesian search, no AutoML,
  no loop over candidate settings inside `train.py`.
- Train and score models only through `evaluate.py`. Do not run your own training or
  validation scripts. Exploring the training data (statistics, plots) is allowed and free.
- No external data, no pretrained models, no internet lookups of labels. Test features are
  for prediction only: do not fit anything on them.
- `train.py` receives its data as arguments and must not read files itself.

**Finish.** When the iteration limit in Part 2 is reached, or earlier if you have clear
evidence that further runs will not help:

1. Declare the final iteration with `python evaluate.py final <N>`.
2. Make sure `results/notes.md` has an entry for **every** iteration, and write
   `results/report.md` using the template in `README.md`. Report honestly, including failed
   iterations and anything a human did for you.
3. Run `python evaluate.py pack` to build `submission.zip`.
4. Stop. Do not run more experiments.
<!-- RULES:END -->

## Part 2 – Research strategy (EDITABLE: participants may rewrite everything below)

### Iteration limit
Iteration limit: 25

### Before the first run (free, not an iteration)
Explore the training data and write a short summary at the top of results/notes.md:
- class balance overall and per well; which facies are rare
- per-well differences in the log distributions (GR, ILD_log10, PE, PHIND, DeltaPHI)
- WELL_02 is synthetic and contains only facies 9: note how it might distort training
- how facies change with depth, and how often neighbouring samples share a facies
- which facies pairs are visually/physically similar

### Loop (run without asking for confirmation)
1. Run `python evaluate.py status`, then read the latest metrics.json: per-well F1,
   per-class F1, confusion matrix.
2. Write one hypothesis, citing the evidence from the previous results that motivates it.
   Make ONE change per iteration so the result is interpretable.
3. Edit train.py and update MODEL_FAMILY if the model changes.
4. `python evaluate.py check`, fix errors, then `python evaluate.py run "<hypothesis>"`.
5. Append to results/notes.md using this exact template:
   - Iter N | Hypothesis | CV weighted F1 | Delta vs best so far | Per-well F1 range |
     Worst class | Keep/Discard | What was learned | Next idea
6. If discarded, restore train.py from the best iteration's snapshot first.

### Decision rules
- A change counts as an improvement only if CV weighted F1 rises by more than 0.01
  AND it does not hurt more than 2 of the 7 real wells noticeably (check per-well F1).
- A gain that comes from only one well is a warning sign: note it, and do not treat it as kept
  without a second run that confirms it.
- Prefer features and models that are simple and physically motivated; with ~3k rows,
  complexity usually overfits.
- All context features (rolling, lag, lead, gradient) must be computed within each well, sorted by
  Depth, using only that well's rows. Never fit scalers or statistics on test rows.
  Per-well normalisation must use only the well's own rows at prediction time and be
  disclosed in the report as a design choice.
- Do not run any training or scoring outside evaluate.py.

### Planned roadmap (iterations 1-12 are done; adapt based on evidence and explain any deviation)
Done so far (best = iteration 8, CV weighted F1 0.5181):
1 baseline RF 0.4870; 2 window-3 context features 0.5014; 3 per-well z-score replacing
absolute values 0.4899 (failed: erased WELL_02's PE); 4 gradients 0.4977; 5 balanced class
weights 0.4819; 6 Formation features 0.4980; 7 LightGBM on iter 2 features 0.4988;
8 per-well z-scores ADDED alongside absolute values 0.5181 (best); 9 RBF SVM 0.4951;
10 equal-weight RF+SVM soft vote 0.5071; 11 MLP 0.4809; 12 probability smoothing along
depth 0.5119.

Next, in this order:
13. NOISE CHECK: rerun the iteration 8 code exactly, changing only SEED to a different
    value. This measures run-to-run spread, so we can tell whether differences of 0.01-0.02
    are real.
14. NOISE CHECK 2: the same, with a third SEED value.
15. LightGBM on the iteration 8 features (iteration 7 used the iteration 2 features, so that
    comparison was confounded).
16. Weighted soft vote, RF-dominant (for example RF 0.7, SVM 0.3), because the equal-weight
    vote in iteration 10 let the SVM overrule the RF on facies 9.
17. Hybrid: SVM predictions only for rows with NM_M = 1 (non-marine, where facies 1 lives and
    facies 9 cannot occur), Random Forest for all other rows. One fixed configuration.
18. XGBoost on the iteration 8 features.
19. Optional: lower sample weight for WELL_02 for the smooth models only, to test whether its
    synthetic facies 9 rows hurt them.
20. Optional: window-based 1D-CNN or LSTM, only if the above all plateau.
Spend runs on the most promising directions, not on tiny hyperparameter nudges.

### Using the noise check
- After iterations 13 and 14, write the three iteration-8 scores (iterations 8, 13, 14) and
  their spread (max minus min) in notes.md.
- If the spread is larger than 0.01, treat any later gain smaller than that spread as noise:
  do not keep it, and say so in the notes.

### Stopping and final choice
- Do not stop early before iteration 14 is finished.
- After iteration 14, stop early if 3 consecutive iterations give no improvement above both
  0.01 AND the measured seed spread over the best score.
- Hard limit: iteration 25.
- Choose the final iteration by best CV score. If another iteration is within the seed spread
  and is simpler or more consistent across wells, prefer it and say why in the report.
  Do not choose by guesswork about the test set.
- At the end: declare the final iteration, complete notes.md for EVERY iteration, update
  report.md, run `python evaluate.py pack`, and stop.

### Resuming this program
- Iterations 1-12 already exist in results/. Run `python evaluate.py status` and read
  results/notes.md first. Do NOT re-run the baseline.
- Start from the best iteration's code: results/iter_08/train.py.
- results/notes.md is append-only. Never edit or delete earlier entries. If you find a mistake
  in an old entry, add a dated correction note at the end. Before writing each entry, check
  that one for that iteration does not already exist.
- Human interventions to record in report.md: (1) after iteration 6, the participant changed
  the stop rule and raised the limit from 20 to 25; (2) after iteration 12, the participant
  reopened the run, which had stopped under the revised rule, and added iterations 13 onwards
  (noise check first). 

### Honesty
Record failed iterations and anything a human changed. Never edit past notes to hide a result.