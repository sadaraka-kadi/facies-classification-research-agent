# IGS 2026 Agentic AI Lab Competition

**Topic: Well Log Facies Classification, AutoResearch style.**
You set up a coding agent; the agent runs a small research program on its own:
propose a hypothesis, change the code, run the experiment, read the result, decide what
to try next. There is no fixed number of iterations: you set the limit for your agent, and
you submit the **10 best iterations** together with complete notes and a report.

The competition looks at two things together:

- **Machine learning skill** – data handling, feature engineering, model choice, experiment design, analysis.
- **Agentic AI skill** – how well you direct an agent to do that research autonomously.

## 1. Task and data

Predict the rock type (facies, 9 classes) at each depth sample from wireline log curves.

| File | Rows | Content |
|---|---|---|
| `data/train_data.csv` | 3,232 | 8 wells, features + `Facies` label |
| `data/test_data.csv` | 809 | 2 other wells, features only. Labels are held by the organizers. |

Columns (rows are depth-ordered within each well, 0.5 ft sampling):

| Column | Meaning |
|---|---|
| `Facies` | Label 1–9, training file only |
| `Formation`, `Well Name`, `Depth` | Stratigraphic unit (`SH` = shale member, `LM` = limestone member), well (coded `WELL_01`…`WELL_10`), measured depth (ft) |
| `GR` | Gamma ray |
| `ILD_log10` | Deep resistivity, log10 |
| `DeltaPHI` | Neutron–density porosity difference |
| `PHIND` | Average neutron–density porosity |
| `PE` | Photoelectric effect |
| `NM_M` | Non-marine (1) / marine (2) indicator |
| `RELPOS` | Relative position within the formation |

Facies: 1 nonmarine sandstone · 2 nonmarine coarse siltstone · 3 nonmarine fine siltstone ·
4 marine siltstone and shale · 5 mudstone · 6 wackestone · 7 dolomite · 8 packstone-grainstone ·
9 phylloid-algal bafflestone. Classes are imbalanced, and neighbouring facies are easily confused.

## 2. Files

```text
README.md          this file
program.md         the prompt for your agent: Part 1 rules are locked, Part 2 is yours to edit
AGENTS.md          auto-loaded by Codex and Cursor; points the agent to program.md
CLAUDE.md          auto-loaded by Claude Code; points to AGENTS.md
train.py           features + model: the only code file you and your agent edit
evaluate.py        official evaluation, iteration counter and recorder (do not modify)
requirements.txt   dependencies (you may add to it)
data/              competition data (do not modify)
results/           created by evaluate.py: one folder per iteration + log.csv
```

## 3. Quick start

Use **Python 3.12** and the package versions pinned in `requirements.txt` (judges rerun your
final code in exactly this environment).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py check                      # free smoke test, prints no score
python evaluate.py run "Baseline: random forest on the 7 raw logs"
python evaluate.py status                     # log and your iteration limit
python evaluate.py final 7                    # declare iteration 7 as your final submission
python evaluate.py pack                       # build submission.zip
```

### Start your agent

Open this folder with your agent and give it one sentence. `AGENTS.md` / `CLAUDE.md` are loaded
automatically and point the agent to `program.md`.

| Agent | How to start |
|---|---|
| Claude Code | `claude "Read program.md and follow it."` |
| Codex CLI | `codex "Read program.md and follow it."` |
| Cursor | File → Open Folder, then in Agent mode type: `Read program.md and follow it.` |
| Any other | Open the folder and give it the same sentence. |

To let the agent work through its runs without stopping at every command, allow it to run
`python evaluate.py` commands automatically in your agent's permission settings. Approving a
command is not a human intervention; changing what the agent decided is, and must be disclosed.

On macOS, XGBoost and LightGBM also need `brew install libomp`.

`program.md` has two parts. **Part 1 (competition rules) is locked**: `evaluate.py` will not run
if it is changed. **Part 2 (research strategy) is yours** to rewrite before you start.

## 4. What one iteration is

`python evaluate.py run "<note>"` does all of this, and it counts as **one** iteration:

1. Snapshots `train.py` into `results/iter_NN/`.
2. **Leave-one-well-out cross-validation** on the training wells: each of the 7 real wells is
   held out once (`WELL_02`, a synthetic well that contains only facies 9, always stays in the training part).
3. Reports the **CV score: weighted F1** over all held-out predictions (plus macro F1,
   accuracy, per-well and per-class F1, confusion matrix).
4. Refits on all training wells and writes predictions for the test wells.
5. Appends one row to `results/log.csv`.

The starter baseline scores about **0.486** CV weighted F1.

Counting rules:

- **No maximum number of iterations**, and no time limit per iteration.
- You set your own limit with the line `Iteration limit: 20` in Part 2 of `program.md`. Change
  the number as you like; `evaluate.py` will not start a run beyond it.
- Every run is logged and numbered, including failed or interrupted runs. The baseline run is
  iteration 1. Use `check` first; it is free.
- One iteration trains **one fixed configuration**. No grid/random/Bayesian search, AutoML,
  or loops over candidate configurations inside `train.py`.
- Reading data, reading earlier results, and `check` / `status` are not iterations.
- Training or scoring models outside `evaluate.py` is not allowed.

## 5. Rules

**You may change**

- `train.py`: features, preprocessing, model, hyperparameters, class-imbalance handling.
  Keep the `fit(train_df)` / `predict(model, df)` interface and keep all modelling code in this one file.
- Part 2 of `program.md`, `AGENTS.md`, `CLAUDE.md`, and `requirements.txt` to uncomment or add packages. Do not change the pinned
  versions; pin any package you add.

**You may not change or do**

- `evaluate.py`, the metric, the validation split, anything in `data/`.
- Part 1 of `program.md` (the block between the `RULES:BEGIN` and `RULES:END` markers).
- Anything already written in `results/` (log, scores, predictions, code snapshots); do not
  delete or reset it.
- Use external data, pretrained models, or the published labels of any well. Use the test
  features only for prediction, never for fitting (no pseudo-labelling, no statistics from test rows).

**Model whitelist:** Random Forest, XGBoost, LightGBM, SVM, MLP, 1D-CNN, LSTM.
An ensemble of whitelisted models is allowed and is one configuration.

**Agent:** any agent product and any LLM. You do not need to build your own agent framework.
Human help is allowed but must be disclosed (see the report template).

## 6. Two scores, one final model

| | CV score | Test score |
|---|---|---|
| Computed on | training wells, leave-one-well-out | the 2 hidden test wells |
| Who sees it | you and your agent, every iteration | judges only, after submission |
| Used for | deciding what to try next and choosing your final iteration | ranking |

Your **final model** is the single iteration you declare with `python evaluate.py final <N>`.
Judges score `results/iter_NN/predictions.csv` of that iteration exactly as archived; there is
no retraining. You can change the declaration any time before you submit. If you declare
nothing, the iteration with the best CV score is used. Judges may rerun your final code.

## 7. Scoring (100 points)

| Part | Points | What judges look at |
|---|---|---|
| Test performance | 50 | Weighted F1 of the final iteration on the hidden test wells, scaled linearly from the starter baseline (0 points) to the best valid submission (50 points). |
| Agentic research process | 25 | Each iteration has a hypothesis and a conclusion; later iterations build on earlier results; the agent works autonomously within the limit you set; the notes cover every iteration and the records are real. |
| ML rigour and insight | 15 | Sound features and validation; error analysis by class and by well; the final choice is justified; CV and test scores are consistent. |
| Report and demo video | 10 | Clear, honest, reproducible. |

A submission is **not ranked** if it modifies `evaluate.py`,
`data/` or past records, uses external data or a model outside the whitelist, or hides
experiments or human intervention.

## 8. What to submit

Send by email to `DSong4@slb.com` before `2026/10/27`:

1. **`submission.zip`, built with `python evaluate.py pack`.** It contains this folder with:
   - the **10 iterations with the best CV score** (code snapshot, output, metrics, predictions),
     plus your final iteration if it is not among them;
   - the **complete** `results/log.csv`, which lists every iteration you ran;
   - `results/notes.md`: one entry for **every** iteration (hypothesis, CV score, keep or
     discard, what was learned), including those outside the top 10;
   - `results/report.md` (template below);
   - `results/agent_log/` with the raw, unedited agent conversation or session export (any format).
2. **A demo video of at most 10 minutes** (file or link). It may be edited for length; show your
   setup, the agent working through several iterations, and the final result.

### `results/report.md` template

```markdown
# Team / participant
# Disclosure
- Agent product and version:
- LLM(s):
- ML model families used:
- Compute (CPU/GPU, RAM):
- Human interventions (iteration, what you did, why): 
# Iteration summary
- Total iterations run, and the iteration limit you set:
| # | Hypothesis / change | CV weighted F1 | Kept? | What was learned |
# Final model
- Declared final iteration and why this one:
- Features, model, key hyperparameters:
# Analysis
- Which facies / wells are still hard, and why:
- What you would try next:
# Agent setup
- Changes made to program.md and what worked or did not work with the agent:
```
