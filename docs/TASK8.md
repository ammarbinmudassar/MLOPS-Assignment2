# Task 8 — Local UI with cloud inference

**Report explanation:** The local Flask app sends uploaded images to the Vercel
API. Results and history are read from the same Neon database.

The ignored `.env.remote` is already configured privately. Start it from the
assignment directory:

```bash
VISIONGUARD_WEB_ENV=.env.remote .venv/bin/python -m task4.run
```

Open http://127.0.0.1:8001. Upload a PNG/JPEG under 3 MiB and click the check
button. Open History to see saved predictions. The local UI needs Internet access
and the API key; it does not load TensorFlow or create a local database in this mode.

For a fresh setup, copy `.env.web.example` to `.env.remote`, set
`INFERENCE_API_URL=https://visionguard-a2-api.vercel.app`, set the private
`INFERENCE_API_KEY`, `MAX_UPLOAD_BYTES=3145728` and `APP_PORT=8001`.
Use `VISIONGUARD_WEB_ENV=.env.remote` to select this file.

## Evidence

| Part | Screenshot |
|---|---|
| Local upload interface | `outputs/task8/screenshots/upload.jpg` |
| Multiple inputs/results | `outputs/task8/screenshots/result_1.jpg` through `result_3.jpg` |
| Persistent history | `outputs/task8/screenshots/history.jpg` |

`outputs/task8/remote_ui_check.json` records the three real UI upload checks.
Different images may receive the same label; preserve the model's actual results.
