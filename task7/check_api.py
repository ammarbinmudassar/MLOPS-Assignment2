"""Exercise the real API without printing credentials or storing image bytes."""
import csv
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / '.env.api', override=False)
    url = os.environ['INFERENCE_API_URL'].rstrip('/')
    headers = {'X-API-Key': os.environ['VISIONGUARD_API_KEY']}
    health = requests.get(url + '/health', timeout=60)
    health.raise_for_status()
    assert health.json()['database_ready']
    unauthorized = requests.get(url + '/history', timeout=60)
    assert unauthorized.status_code == 401
    rows = list(csv.DictReader((ROOT / 'outputs/task2/splits/validation.csv').open()))
    chosen = [next(row for row in rows if row['class'] == label) for label in ('AI', 'Real')]
    chosen.append(next(row for row in rows if row['class'] == 'AI' and row != chosen[0]))
    report = {'url': url, 'health': health.json(), 'unauthorized_history_status': 401, 'predictions': []}
    for index, row in enumerate(chosen, 1):
        path = ROOT / 'dataset' / row['path']
        with path.open('rb') as image:
            result = requests.post(url + '/predict', headers=headers,
                                   files={'image': (f'validation_{index}{path.suffix}', image)}, timeout=60)
        assert result.status_code == 201, result.status_code
        record = result.json()
        fetched = requests.get(url + '/results/' + record['id'], headers=headers, timeout=60)
        fetched.raise_for_status()
        assert fetched.json()['id'] == record['id']
        report['predictions'].append({'dataset_label': row['class'], 'response': record})
    history = requests.get(url + '/history', headers=headers, timeout=60)
    history.raise_for_status()
    identifiers = {record['id'] for record in history.json()['results']}
    assert all(item['response']['id'] in identifiers for item in report['predictions'])
    report['history_verified'] = True
    (ROOT / 'outputs/task7/public_api_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Public API: health, authentication, three predictions, result lookup and Neon history passed.')


if __name__ == '__main__':
    main()
