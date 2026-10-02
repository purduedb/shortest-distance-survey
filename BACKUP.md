# Backup

`~/scratch` is purged after 60 days of inactivity and is never backed up. Git covers all source/config/docs — this only applies to large git-ignored directories (`data/`, `non_ml_index/`, `results/`, `claude_work/`), which need a copy on [Fortress](https://docs.rcac.purdue.edu/userguides/fortress/), Purdue's tape/disk archive. `hsi`/`htar` are on `PATH` by default on Gilbreth (front-end and compute nodes), no module load needed.

## Layout

- Fortress: `/home/$USER/shortest-distance-survey/`
- Local staging: `~/scratch/backups/shortest-distance-survey/` (outside the repo, so tars never end up inside other tars)

Names are `<yyyy-mm-dd>-<what>.tar`; each tar extracts to the original directory name.

| Tar | Contents | Extract into |
|-----|----------|--------------|
| `<date>-results-<expt>.tar` | full `results/<expt>/` | `results/` |
| `<date>-results-<expt>-lite.tar` | same, minus `saved_models/`, `saved_jit_models/`, `saved_onnx_models/` (metrics, logs, jobs, plots — enough for `analysis/`) | `results/` |
| `<date>-data.tar` | full `data/` | repo root |
| `<date>-data-lite.tar` | `data/README.md`, `*.edges`, `*.nodes`, `stats.json`, `real_workload/` queries | repo root |
| `<date>-claude_work.tar` | full `claude_work/` | repo root |
| `<date>-non_ml_index.tar` | full `non_ml_index/` (HC2L indexes) | repo root |

After upload, only the `-lite` tars are kept in local staging. A `<tar>.filelist.txt` manifest (`tar tvf` output) for every tar is kept there too — use it to find what's inside a tar without recalling it from tape.

## Backing up

Plain `tar cf` (no compression; contents are mostly binary). Tar locally first, then transfer — `htar` (which tars during transfer) has a 64GB per-file limit this repo's checkpoints exceed. Build one tar at a time, then upload one at a time — tar and `hsi put` compete for scratch I/O (~390 MB/s uncontended vs. ~59 MB/s contended). Run on an interactive job (`srun --jobid <id> --overlap ...`) rather than a login node.

```bash
D=$(date +%F); B=~/scratch/backups/shortest-distance-survey; R=/home/$USER/shortest-distance-survey
mkdir -p $B && cd ~/scratch/shortest-distance-survey

# Build
(cd results && tar cf $B/$D-results-v9-camera-training.tar v9-camera-training)
(cd results && tar cf $B/$D-results-v9-camera-training-lite.tar \
    --exclude=v9-camera-training/saved_models --exclude=v9-camera-training/saved_jit_models \
    --exclude=v9-camera-training/saved_onnx_models v9-camera-training)
tar cf $B/$D-data.tar data
{ echo data/README.md; find data -mindepth 2 -maxdepth 2 \( -name '*.edges' -o -name '*.nodes' -o -name stats.json \); \
  find data -mindepth 2 -maxdepth 2 -type d -name real_workload; } | sort > $B/$D-data-lite.inputlist.txt
tar cf $B/$D-data-lite.tar -T $B/$D-data-lite.inputlist.txt
tar cf $B/$D-claude_work.tar claude_work
tar cf $B/$D-non_ml_index.tar non_ml_index
for t in $B/$D-*.tar; do tar tvf $t > $t.filelist.txt; done

# Upload
hsi mkdir -p $R
for t in $B/$D-*.tar; do hsi "put $t : $R/$(basename $t)"; done
```

## Verifying

```bash
hsi "ls -l /home/$USER/shortest-distance-survey/"   # byte counts must match `ls -l` of the local tars
```

Once sizes match, delete the full (non-lite) local tars.

## Restoring

```bash
B=~/scratch/backups/shortest-distance-survey; R=/home/$USER/shortest-distance-survey; T=<date>-<what>.tar
hsi "get $B/$T : $R/$T"
tar xf $B/$T -C ~/scratch/shortest-distance-survey/          # results-* tars: add /results
tar xf $B/$T -C ~/scratch/shortest-distance-survey/ <path>   # single file/dir (path as listed in the manifest)
```

## Common `hsi` commands

```bash
hsi ls -l /home/$USER/shortest-distance-survey/      # list a directory
hsi "cd /home/$USER; find . -name '*.tar'"           # search recursively by pattern
hsi "put local : remote"                             # upload
hsi "get local : remote"                             # download (local path first, same as put)
hsi mkdir -p /home/$USER/<dir>                       # create a directory
hsi "rm /home/$USER/file.tar"                        # delete
hsi "mv old new"                                     # rename/move
hsi du                                               # space usage on Fortress
```
