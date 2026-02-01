## Steps to Submit SLURM Jobs on Gilbreth Cluster (Purdue RCAC)

* Basic Job Submission
```bash
sbatch basic_job.sh
```

* Single Job Submission
```bash
sbatch job.sh
```

* Multiple Job Submission (using script)
```bash
bash urban_expt.sh --execute --slurm
```

* Interactive job:
```bash
sinteractive -N 1 -n 16 --gres=gpu:1 --partition=v100 --mem=32G --account=csml --qos standby --time 120
```

* Notes
    * Use `tmux` (usage: tmux new -s one or tmux a -t one) for persistent sessions. Note the front-end node because tmux session is node specific.
    * You may also use `screen` (usage: screen -S one or screen -r one) for persistent sessions. Note the front-end node because screen session is node specific.
    * Using `array` jobs or `dependent` jobs didn't work and gave errors (`sbatch: error: QOSMaxSubmitJobPerUserLimit`)
