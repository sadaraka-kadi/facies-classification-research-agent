# Using an AI agent to run its own machine learning research: a well log case study

A personal re-run of the **IGS 2026 Agentic AI Lab Competition** (well log facies classification), in which a coding agent, **Claude Code (Opus 5.5)**, proposed, ran and evaluated 20 experiments on its own. I wrote the instructions, set the stop rules and reviewed the results.

Talk: *Using an AI Agent to Run Its Own Machine Learning Research*, 26 November 2026.
Author: Kadi Sadaraka, Point Ensemble.

## The task

Predict the rock type (facies, 9 classes) at each depth sample from 7 wireline log inputs: `GR`, `ILD_log10`, `DeltaPHI`, `PHIND`, `PE`, `NM_M`, `RELPOS`.

- Training data: 8 wells, 3,232 rows (one of them, WELL_02, is synthetic and contains only facies 9).
- Test data: 2 wells, 809 rows, labels held by the organisers.
- Validation: leave-one-well-out cross-validation on the 7 real wells. The score is weighted F1, the competition metric.

## Result

| | CV weighted F1 |
|---|---|
| Starter baseline (random forest, 7 raw logs), iteration 1 | 0.487 |
| Final model, iteration 17 (LightGBM) | **0.5365** |

Iteration 17: macro F1 0.495, accuracy 0.549.

Read these numbers with care:

- **The 0.5365 is the best of 20 runs**, so it is optimistic. Re-running with different random seeds moved the score by about 0.01 F1, so differences smaller than that are not meaningful.
- **There is no test score.** The test labels are not public and I did not submit.
- The model is still weak on some classes. Mudstone (facies 5) has F1 of about 0.17, and the largest confusion is between facies 2 and 3.

## What the agent found

- Adding **context features** (rolling mean and standard deviation, and the previous and next sample, for each curve) helped, because neighbouring samples usually share the same rock type.
- **Per-well normalisation** of the curves helped, but it nearly erased the signal in WELL_02, a well with a single facies. Normalising WELL_02 with statistics from the other training wells fixed this and gave the biggest single gain.
- **LightGBM** beat random forest once the features were in place.
- Balanced class weights, SVM, a small neural network and several ensembles did not help.

## What is in this repository

```
train.py            final training script (iteration 17 snapshot is in results/)
requirements.txt    package versions
instructions/       the four versions of the agent's instructions (see below)
results/
  log.csv           one row per iteration: score, model family, note
  final.json        the declared final iteration
  notes.md          the agent's notes on every iteration
  report.md         the agent's report
  iter_NN/          code snapshot, console output and metrics for each run
scripts/            helper scripts used for the talk's charts and checks
```

### The instructions

The agent follows a file called `program.md`. Part 1 of that file is the competition's locked rules. Part 2 is the research strategy, which I edited. I rewrote it three times during the project:

1. `1_program_original.md`: the starter version from the kit.
2. `2_experiment_one.md`: a 20-iteration limit and a stop rule of 4 consecutive runs with no gain.
3. `3_experiment_two.md`: a 25-iteration limit, instructions to try boosting, SVM/MLP and an ensemble, and a roadmap of ideas.
4. `4_program_final.md`: iterations 1 to 12 recorded, noise checks first, and a stricter stop rule.

A change counted as an improvement only if CV F1 rose by more than 0.01 and it did not hurt more than 2 of the 7 wells.

## What is not included

- **The competition data** (`data/train_data.csv`, `data/test_data.csv`). The organisers asked that it not be shared publicly. Get it from the IGS 2026 Agentic AI Lab Competition kit.
- **`evaluate.py`**, the official evaluation harness. It is part of the kit, not my work.
- **Test-set predictions and any submission zip.**

## Reproducing the run

1. Get the IGS 2026 Agentic AI kit from the organisers and set up Python 3.12 with the packages in `requirements.txt`.
2. Copy `train.py` from this repository into the kit folder, replacing the starter file.
3. Run `python evaluate.py run "note"` to reproduce a score (about 0.5365 for this model, give or take the seed noise).

## Limitations

- One agent, one run through 20 iterations, one dataset. The results show what happened in this run, not a general rule.
- Only 7 real wells in the cross-validation, so per-well and per-class numbers are noisy.
- Not every whitelisted model was tried. XGBoost, 1D-CNN and LSTM were left out, and I have listed them as next steps.
- I directed the agent: the stop rules and roadmaps were mine, and the agent chose its own ideas for some iterations (12 and 17 to 20).

## Credits

Data and evaluation harness: IGS 2026 Agentic AI Lab Competition organisers (SLB). Agent: Claude Code with Claude Opus 5.5.

## Licence

MIT Licence
