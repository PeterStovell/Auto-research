# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For general information about the repo, see README.md.

## Objective

Our objective is to improve on our current Pytorch model (see `train.py`) and lower the validation loss.

## Experimentation

**What you CAN do:**
- Inside `train.py` modify everything except the function `compute_eval_loss`
- Modify `features.py`

**What you CANNOT do:**
- Install new packages or add dependencies. 
- Modify the evaluation function `compute_eval_loss` in `train.py`. The `compute_eval_loss` function in `train.py` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** 

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.


## The experiment loop

LOOP FOREVER:

1. Implement an experimental idea by directly changing the code
2. Pick a short experiment name e.g. "tcn_bz256"
3. Launch the experiment on kubeflow using the experiment name
4. Wait for the run using wait_kfp.sh {run_id}
5. When done, read the run_summary.yaml. Append a row to EXPERIMENTS_TIMELINE.md with the run number, val_loss, experiment name, and a short summary. Mark with ⭐ if it is an improvement.
6. If the experiment is an improvement, commit it. If the experiment is a fail try to fix it or roll back to the latest commit and start something new.

The first experiment should be the baseline (name: baseline, no changes to the repo).

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. 

Run experiments in sequence, learning from the previous experiment. Commit only the improvements.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
