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

This is a starting point. Change it to make your agent work better.

### Iteration limit

Iteration limit: 20

Change the number above to give your agent more or fewer runs (delete the line for no limit).
`evaluate.py` reads it and refuses to start a run beyond it.

### Before the first run

Look at the training data: class balance, per-well differences, how facies change with depth.

### Loop

Repeat until the iteration limit is reached, without asking for confirmation:

1. Run `python evaluate.py status` and read the latest `results/iter_NN/metrics.json`
   (per-well F1, per-class F1, confusion matrix).
2. Write down **one hypothesis**: what you will change, and why the evidence so far
   suggests it will help. Prefer one change per iteration so the result is interpretable.
3. Edit `train.py`. Update `MODEL_FAMILY` if the model changes.
4. Run `python evaluate.py check` and fix any error.
5. Run `python evaluate.py run "<hypothesis in one sentence>"`.
6. Append to `results/notes.md`: iteration number, hypothesis, CV score, the change versus the
   best score so far, keep or discard, and what you learned.
7. If the change did not help, restore `train.py` from the best iteration's snapshot
   (`results/iter_NN/train.py`) before the next idea.

### Guidance

- Spend runs deliberately. Differences smaller than about 0.01 between two runs are likely
  noise, and the more runs you compare, the more the best CV score flatters itself.
- Watch per-well scores: a change that helps one well and hurts the others is unlikely to
  transfer to new wells.
- Normally choose the iteration with the best CV score as final; say why if you choose another.

### Ideas, in no particular order

Per-well depth context (gradients, rolling windows, neighbouring samples), per-well
normalisation, using `NM_M` and `Formation`, class weights, tree boosting versus neural
sequence models, a small ensemble near the end.
