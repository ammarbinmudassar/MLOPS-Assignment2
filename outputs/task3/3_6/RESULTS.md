# Task 3.6 results

Registered model: `23L-0630_VisionGuard`.

| Version | Configuration | Status | Alias |
|---|---|---|---|
| 1 | unfreeze_4 | READY | |
| 2 | unfreeze_8 | READY | champion |

The separate `task3/predict_champion.py` script loaded
`models:/23L-0630_VisionGuard@champion`, resolved version 2 and classified the
first saved test image as AI with 55.39% raw confidence. Its dataset label is AI.
The alias-loaded model and local champion produced matching predictions when
checked with the same single-image batch. No model was retrained.

For evidence, capture the registry showing both versions and the champion alias.
Run `cat outputs/task3/3_6/registration.txt` and
`cat outputs/task3/3_6/prediction.txt` for clean terminal evidence.

Short explanation: I registered two model versions and assigned champion to the
validation-selected winner. A separate script loads this alias and classifies
one image. Deployment will download the chosen version once and load it locally
at startup, avoiding DagsHub calls for individual requests.
