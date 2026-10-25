# Task 6.6 — GitHub Actions CI

## Project workflow

`.github/workflows/ci.yml` runs after pushes and pull requests, and supports manual
runs. It installs CPU-only CI dependencies, runs Ruff and all unit tests, then
builds the Docker image. It requires no real credentials, dataset images or
champion download. The split manifests used by the audit test are tracked in Git.
The README contains the normal GitHub Actions status badge.

Local checks: Ruff passed; all 29 tests passed. Hosted evidence is pending GitHub
sign-in and publication; a local pass is not a hosted green run.

## Toy workflow

The toy template is in `outputs/task6/6_6/toy/`. Its workflow runs after each push
on Python 3.11 and 3.12, with fail-fast disabled so both jobs complete.
For the intentional red commit, `add` returns `left - right`. The fix restores
`left + right`; the same test then passes. Each revision must be pushed separately
and allowed to finish so both red and green runs remain available as evidence.

The toy repository is separate from the assignment repository.

## Evidence to submit

- Toy failed run showing both Python matrix jobs.
- Toy successful run showing both Python matrix jobs.
- Successful VisionGuard run showing dependency installation, lint, tests and image build.
- README badge linked to the project workflow.

Only real GitHub Actions runs count as hosted evidence. No simulated screenshots
or HTML evidence sheets are used.

## Short secrets explanation

Store credentials in repository Settings → Secrets and variables → Actions,
then reference them through the workflow's `secrets` context. Never put values in
YAML, because committed files remain visible in repository history. This CI uses
mock credentials and needs no project secrets.

Sources: [Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and [Actions secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).
