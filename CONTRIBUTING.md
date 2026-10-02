# Contributing

## Branch workflow

All work happens on `dev`. `main` only ever moves forward via fast-forward merge — never commit directly to `main`.

1. Do all work/commits on `dev` only. If `dev` doesn't exist yet, create it once with `git checkout -b dev` and push with `git push -u origin dev`; otherwise just `git checkout dev`.
2. If there are uncommitted changes, `git stash push -u` first, `git stash pop` after (step 2 below).
3. When ready to publish to `main`:
   ```bash
   git checkout main
   git merge --ff-only dev
   git push origin main
   git checkout dev
   ```
4. Keep committing on `dev`. Repeat step 3 whenever you want `main` updated.

This stays conflict-free forever as long as `main` is never committed to directly — `--ff-only` will never fail since `main` can never diverge from `dev`.
