#!/usr/bin/env bash
# Usage: bash ~/scratch/shortest-distance-survey/tmux-init.sh
#
# Creates a tmux session for the shortest-distance-survey project.
# Skips creation if the session already exists.

SESSION="one"
WORKDIR="$HOME/scratch/shortest-distance-survey"
CONDA_ENV="myenv"

# ── Guard: skip if session already exists ────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists — attaching."
    tmux attach-session -t "$SESSION"
    exit 0
fi

# ── Create session with first window (RESOURCES) ─────────────────────────────
## TOP PANE: watch -d -c -n20 myshowpartitions
tmux new-session -d -s  "$SESSION" -n RESOURCES -c "$WORKDIR"
tmux send-keys -t       "${SESSION}:RESOURCES" "watch -d -c -n20 myshowpartitions" Enter
## MIDDLE PANE: slist
tmux split-window -v -t "${SESSION}:RESOURCES" -c "$WORKDIR"
tmux send-keys -t       "${SESSION}:RESOURCES" "slist" Enter
## BOTTOM PANE: sinfo | grep idle
tmux split-window -v -t "${SESSION}:RESOURCES" -c "$WORKDIR"
tmux send-keys -t       "${SESSION}:RESOURCES" "sinfo | grep idle" Enter

# ── JOBS window ───────────────────────────────────────────────────────────────
## TOP PANE: watch -d -n20 squeue --me --long
tmux new-window -t      "$SESSION" -n JOBS -c "$WORKDIR"
tmux send-keys -t       "${SESSION}:JOBS" "sleep 5" Enter
tmux send-keys -t       "${SESSION}:JOBS" "watch -d -n20 squeue --me --long" Enter
## BOTTOM PANE: slurm-jobs directory
tmux split-window -v -t "${SESSION}:JOBS" -c "$WORKDIR/slurm-jobs"

# ── SINTERACTIVE window ───────────────────────────────────────────────────────
tmux new-window -t      "$SESSION" -n SINTERACTIVE -c "$WORKDIR"
tmux send-keys -t       "${SESSION}:SINTERACTIVE" "module load conda && conda activate $CONDA_ENV" Enter
tmux send-keys -t       "${SESSION}:SINTERACTIVE" "sinteractive -N1 -n1 -c16 --gres=gpu:1 --partition=training --mem=128G --account=csit --qos training --time 12:00:00"

# ── CLAUDE window ──────────────────────────────────────────────────────────────
tmux new-window -t      "$SESSION" -n CLAUDE -c "$WORKDIR"

# ── Focus RESOURCES on attach ─────────────────────────────────────────────────
tmux select-window -t   "${SESSION}:RESOURCES"

echo "tmux session '$SESSION' created — attaching."
tmux attach-session -t "$SESSION"
