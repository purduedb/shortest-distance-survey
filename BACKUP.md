# Backup

`~/scratch` is purged after 60 days of inactivity and is never backed up. Git covers all source/config/docs — this only applies to large git-ignored directories (`data/`, `non_ml_index/`, `results/`, `claude_work/`), which need a copy on [Fortress](https://docs.rcac.purdue.edu/userguides/fortress/), Purdue's tape/disk archive. `hsi`/`htar` are on `PATH` by default on Gilbreth, no module load needed.

## Backing up

Tar locally first, then transfer — Fortress is tape-backed, so one large stream beats thousands of loose files, and `htar` (which tars during transfer) has a 64GB per-file limit this repo's checkpoints can exceed. One tar per top-level directory keeps individual files recoverable later without pulling a multi-hundred-GB archive.

```bash
mkdir -p ~/scratch/backups
cd ~/scratch/shortest-distance-survey

tar cf ~/scratch/backups/shortest-distance-survey-scratch-<dir>-$(date +%Y%m%d).tar <dir>
hsi "put ~/scratch/backups/shortest-distance-survey-scratch-<dir>-$(date +%Y%m%d).tar : /home/$USER/shortest-distance-survey-scratch-<dir>-$(date +%Y%m%d).tar"
```

Repeat per directory. Avoid building the next tar while a transfer is still running — both compete for scratch I/O (observed ~260-480 MB/s uncontended vs. ~59 MB/s contended).

## Verifying

```bash
hsi "ls -l /home/$USER/shortest-distance-survey-scratch-<dir>-<date>.tar"   # confirm byte count matches the local file
```

## Restoring

```bash
hsi "get /home/$USER/shortest-distance-survey-scratch-<dir>-<date>.tar : ~/scratch/backups/<same-name>.tar"
tar xf ~/scratch/backups/<same-name>.tar -C ~/scratch/shortest-distance-survey/
```

## Common `hsi` commands

```bash
hsi ls -l /home/$USER/                               # list a directory
hsi "cd /home/$USER; find . -name '*.tar'"           # search recursively by pattern
hsi "put local : remote"                             # upload
hsi "get remote : local"                             # download
hsi "mput -R dirname"                                # recursive upload
hsi "rm /home/$USER/file.tar"                        # delete
hsi "mv old new"                                     # rename/move
hsi du                                               # space usage on Fortress
```
