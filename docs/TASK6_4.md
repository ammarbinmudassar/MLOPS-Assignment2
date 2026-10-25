# Task 6.4 — image analysis

| Build | Docker-reported image size |
| --- | --- |
| Before (`visionguard:6.4-before`) | 2,195,762,884 bytes — 2.196 GB |
| After (`visionguard:6.4-after`) | 2,073,027,979 bytes — 2.073 GB |

Reduction: **122,734,905 bytes (122.7 MB), or 5.59%**. GB and MB use decimal units.
These measurements use `docker image inspect .Size`, not a registry download size
or the combined disk usage of images sharing layers.

## Short report explanation

Adding `--no-compile` to pip installation reduced the image from 2.196 GB to
2.073 GB, saving 122.7 MB (5.59%). The TensorFlow/runtime dependency layer remains
the largest at 1.514 GB, followed by Debian (86.7 MB) and Python (47.4 MB).

Pip normally compiles installed Python source into bytecode; this option skips
that step. The application already disables writing bytecode at runtime.
Imports may take longer without precompiled bytecode; no startup speed claim is made.
[Official pip options](https://pip.pypa.io/en/stable/cli/pip_install/).

## Commands for terminal screenshots

```bash
docker image inspect visionguard:6.4-before visionguard:6.4-after --format '{{.RepoTags}}: {{.Size}} bytes'
```

```bash
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' visionguard:6.4-after | head -n 12
```

Submit the two size numbers and short explanation above. The terminal screenshots
show the measurement and largest layers; no HTML images were created.

## Verification and reproducibility

The Dockerfile changed only by adding `--no-compile` to its two pip installation
commands. The base image, source files and installed package versions were the
same. The earlier Dockerfile is saved at `outputs/task6/6_4/Dockerfile.before`.
Rebuild it with:

```bash
docker build -f outputs/task6/6_4/Dockerfile.before -t visionguard:6.4-before .
docker build -t visionguard:6.4-after .
```

Existing tags already preserve both measured builds. Future dependency releases
can affect fresh builds because transitive requirements are not fully locked;
`packages_before.txt` and `packages_after.txt` record the identical versions in
this comparison.

Compose now uses the optimized image. The app remains healthy at
http://127.0.0.1:8083, PostgreSQL history works, and a real-image prediction matches
the baseline saved probability. Results, layer sizes and build output are under
`outputs/task6/6_4/`.
