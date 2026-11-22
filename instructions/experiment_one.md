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
Iteration limit: 20

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

### Planned roadmap (adapt based on evidence, but explain any deviation in notes)
1. Baseline RF on the 7 raw logs.
2. Rolling mean/std and lag/lead features within each well (small window first).
3. Per-well normalisation of the logs (z-score or rank within well).
4. Depth gradients / differences between neighbouring samples.
5. Class weights (balanced) to help rare facies.
6. Domain features (e.g. GR/ILD/PE combinations, Formation x NM_M interactions, RELPOS).
7. LightGBM or XGBoost on the best feature set.
8. SVM or MLP on scaled features.
9. Treat WELL_02 differently (e.g. sample weight) if error analysis shows it hurts.
10. Targeted fix for the worst class pair from the confusion matrix.
11. Simple ensemble of the two best distinct model families.
12. Optional: window-based 1D-CNN or LSTM if tree models plateau.
Then use remaining runs for the most promising directions, not for tiny hyperparameter nudges.

### Stopping and final choice
- Stop early if 4 consecutive iterations give no improvement above 0.01.
- Choose the final iteration by best CV score, unless a slightly lower one is clearly more
  consistent across wells; if so, say why in the report. Do not choose by guesswork about the test set.
- At the end: declare the final iteration, complete notes.md for EVERY iteration,
  write report.md from the README template, run `python evaluate.py pack`, and stop.

### Honesty
Record failed iterations and anything a human changed. Never edit past notes to hide a result.