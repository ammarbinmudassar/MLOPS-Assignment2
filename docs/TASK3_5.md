# Task 3.5 — six controlled experiments

Code: `task3/compare_experiments.py`.

All runs use MobileNetV2 ImageNet weights, the same saved split, seed 42,
160-pixel images, batch size 32, and up to 8 head-training plus 8 adaptation
epochs. BatchNorm stays frozen. Early stopping uses validation loss with
patience 3. The best checkpoint from either phase is retained.

| Run | Single change from reference |
|---|---|
| reference | Frozen backbone, adaptation LR 0.0001, augmentation on, dropout 0.3 |
| lower_lr | Adaptation LR 0.00003 |
| no_augmentation | Augmentation off |
| higher_dropout | Dropout 0.5 |
| unfreeze_4 | Unfreeze the last 4 backbone layers, keeping BatchNorm frozen |
| unfreeze_8 | Unfreeze the last 8 backbone layers, keeping BatchNorm frozen |

The head phase always uses learning rate 0.001. All five variants change just
one reference setting. The training budget and selection rule were committed
before training. The learning rate in the comparison table is the adaptation LR.

Selection uses the lowest validation binary cross-entropy, with validation
ROC-AUC breaking ties. Test data does not determine this choice. Only the chosen
run receives final test evaluation, at the fixed 0.5 threshold.

## Evidence

On DagsHub, open experiment `23L-0630_A2_Ops`, filter to runs tagged `task=3.5`
with status **FINISHED**, select the six `3.5_...` runs and choose **Compare**. Show parameters
`learning_rate`, `augmentation`, `dropout`, `unfrozen_layers`, and metrics
`selection_val_loss`, `selection_val_roc_auc`, `selection_val_accuracy` and
`selection_val_f1`. Capture the comparison view.

`outputs/task3/3_5/comparison.png` is the saved comparison chart.
`outputs/task3/3_5/RESULTS.md` contains the measured results and short justification
after successful completion. Each remote run also receives the comparison files.

To resume interrupted work without retraining finished runs:

```bash
bash run_gpu.sh python -u -m task3.compare_experiments
```

Use `--run higher_dropout` (or another configuration name) to execute one run at
a time. The comparison finalizes automatically once all six results exist.
One interrupted no-augmentation attempt was retained as KILLED; its completed
retry is used in the six-run comparison.
