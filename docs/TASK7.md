# Task 7 — Hosted API

**Report explanation:** The float16 model runs on Vercel. Predictions are saved
in Neon PostgreSQL; an API key protects prediction and history requests.

API URL: https://visionguard-a2-api.vercel.app

## Deployment steps

1. Create a free Neon PostgreSQL project. Save its pooled SSL connection URL
   privately as `DATABASE_URL`.
2. Use `deploy/vercel_api/` as the Vercel project directory. It includes the
   quantized model, Python 3.12 setting and inference dependencies.
3. Run these commands from the assignment directory:

```bash
npx vercel login
npx vercel link --cwd deploy/vercel_api
npx vercel env add DATABASE_URL production --cwd deploy/vercel_api
npx vercel env add VISIONGUARD_API_KEY production --cwd deploy/vercel_api
npx vercel deploy --prod --cwd deploy/vercel_api
curl https://visionguard-a2-api.vercel.app/health
```

Enter secrets privately at the prompts. Never commit `.env*`, `.vercel/`, API keys
or database passwords. The existing deployment already has its secrets configured.
Redeploy after changing production variables. This deployment uses the CLI;
automatic deployment on GitHub pushes is not connected.

## Postman

Import `postman/VisionGuard.postman_collection.json`. Set `api_key` privately
using the value in the ignored `.env.api`; `base_url` is already populated.
For **Predict**, choose Body → form-data → `image` → File → a PNG/JPEG under 3 MiB.
Click Send: expect **201**, label, confidence, model version and result ID.
Then send **Saved result** and **History**: expect **200**. The prediction test
automatically saves `result_id`. **Unauthorized history** should return **401**.
Repeat Predict with other images. Do not export or screenshot the key.

The same collection was checked using Newman; its real terminal output is saved
in `outputs/task7/postman_newman.txt`. A Postman GUI screenshot must be captured
in Postman after importing it.

## Model check

Float16 reduces the model from 15.44 MB to 4.52 MB. On the same 146 validation
images, accuracy changed from 82.88% to 82.19%; 97.95% of decisions agreed.
The deployed image decoder/resize also differs slightly from TensorFlow.
The original champion and registry alias remain unchanged; the derivative is
identified as `2-fp16`. These are validation results, not a new test evaluation.
Confidence is an uncalibrated model score.

## Evidence

| Evidence | File/view |
|---|---|
| Neon project | `outputs/task7/screenshots/neon_project.jpg` |
| Vercel build | `outputs/task7/screenshots/vercel_ready.jpg`, `build_inspect.txt` |
| Live API health | `outputs/task7/screenshots/public_health.jpg` |
| Three predictions + persisted history | `outputs/task7/public_api_check.json` |
| Quantization validation | `outputs/task7/validation.json` |
| Postman collection tests | `outputs/task7/postman_newman.txt` |
| Another network | [Successful GitHub runner check](https://github.com/ammarbinmudassar/MLOPS-Assignment2/actions/runs/42481179553), `outputs/task7/external_network_job.log` |

The hosted check tests public health and authentication from a GitHub runner.
Workstation prediction tests are separate. If the rubric requires prediction
from another network, repeat the Postman prediction using a mobile hotspot and
capture the response.
