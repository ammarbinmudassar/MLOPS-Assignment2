"""Compare on validation only; keep quantization and preprocessing effects separate."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from task4.inference import VisionGuardPredictor
from visionguard.preprocessing import decode_resize_bytes
from deploy.vercel_api.lite_model import LitePredictor

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = VisionGuardPredictor(ROOT / 'models/champion')
    lite = LitePredictor()
    rows = list(csv.DictReader((ROOT / 'outputs/task2/splits/validation.csv').open()))
    labels, baseline, quantized, deployed = [], [], [], []
    for row in rows:
        contents = (ROOT / 'dataset' / row['path']).read_bytes()
        labels.append(int(row['label']))
        baseline.append(original.predict_bytes(contents)['p_ai'])
        pixels = decode_resize_bytes(contents, 160).numpy().astype('float32')[None]
        quantized.append(lite.predict_array(pixels))
        deployed.append(lite.predict_bytes(contents)['p_ai'])
    base = np.array(baseline)
    report = {'split': 'validation only', 'count': len(rows), 'threshold': .5,
              'preprocessing_note': 'Pillow RGB decode plus NumPy antialiased bilinear resize/uint8 letterbox. JPEG decoder differences are measured, not claimed identical.'}
    for name, probabilities in [('keras', baseline), ('lite_same_tf_pixels', quantized), ('lite_pillow', deployed)]:
        values = np.array(probabilities)
        report[name] = {'accuracy': accuracy_score(labels, values >= .5),
                        'roc_auc': roc_auc_score(labels, values),
                        'mean_absolute_probability_drift': float(np.abs(values - base).mean()),
                        'max_probability_drift': float(np.abs(values - base).max()),
                        'decision_agreement': float(((values >= .5) == (base >= .5)).mean())}
    accepted = (report['keras']['accuracy'] - report['lite_pillow']['accuracy'] <= .01
                and report['lite_pillow']['decision_agreement'] >= .97
                and report['lite_same_tf_pixels']['mean_absolute_probability_drift'] <= .005)
    report['acceptance_gate'] = {'max_accuracy_drop': .01, 'min_decision_agreement': .97,
                                 'max_quantization_mae': .005, 'passed': bool(accepted)}
    (ROOT / 'outputs/task7/validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit('Deployment derivative failed the validation gate')


if __name__ == '__main__':
    main()
