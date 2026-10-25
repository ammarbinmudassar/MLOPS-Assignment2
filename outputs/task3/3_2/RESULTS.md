# Task 3.2 results

Remote repository: https://dagshub.com/ammarbinmudassar/MLOPS-Assignment2

Experiment: `23L-0630_A2_Ops`

| Learning rate | Final validation accuracy | Final validation loss |
|---|---:|---:|
| 0.001 | 84.25% | 0.4531 |
| 0.0001 | 78.60% | 0.6533 |

Both runs used the same Fashion-MNIST split, seed, model and five epochs.
Each saved parameters, four metrics per epoch, training curves, history and a
Keras model. Each remote model was loaded back and its predictions verified.

`connection.json` records the authenticated connection check. `verification.json`
records remote run and artifact checks. `gitignore_proof.txt` shows that `.env`
is ignored; the token is not included in these results.

Report explanation: I used DagsHub's hosted MLflow server to track two runs.
Learning rate 0.001 gave the higher validation accuracy. Credentials are loaded
from an ignored `.env` file.

Screenshots to capture manually:

1. DagsHub project and its linked GitHub repository.
2. Experiment comparison showing both finished runs and their metrics.
3. Run artifacts showing `plots/training_curves.png` and the `model` folder.
4. Terminal output of `git check-ignore -v .env`.

The GitHub connection was supplied by the user; browser screenshots were not
captured because the browser pane requires a separate sign-in.
