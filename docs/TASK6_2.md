# Task 6.2 — web app Dockerfile

## Short report explanation

The slim Python base and CPU-only TensorFlow keep out unnecessary GPU libraries.
Dependencies are installed before copying source, so code edits can reuse Docker's
dependency cache. The app runs as user 10001, includes a health check and excludes
credentials through an allowlist `.dockerignore`.

The prepared champion model is mounted read-only at runtime, rather than copied
into the image. A named volume stores SQLite history outside the container.

Sources: [Docker build best practices](https://docs.docker.com/build/building/best-practices/)
and [TensorFlow CPU package](https://www.tensorflow.org/install/pip).

## Commands in order

Run from the assignment folder. Port 8082 is separate from the existing local app.

```bash
docker build -t visionguard:6.2 .
```

```bash
docker run -d --name visionguard-62-web \
  -p 127.0.0.1:8082:8000 \
  --mount "type=bind,source=$(pwd)/models/champion,target=/models/champion,readonly" \
  --mount source=visionguard-62-data,target=/data \
  visionguard:6.2
```

If the demonstration container already exists, use the commands below directly.
Do not run a second container with the same name.

```bash
docker exec visionguard-62-web id
```

```bash
docker inspect visionguard-62-web --format '{{.State.Health.Status}}'
curl http://127.0.0.1:8082/health
```

The health state initially says `starting`; rerun after startup. Open
http://127.0.0.1:8082 to try the app. Read startup errors with:

```bash
docker logs --tail 30 visionguard-62-web
```

## Evidence to submit

- Link to `Dockerfile` and `.dockerignore` in your repository.
- Terminal screenshot of `docker exec visionguard-62-web id`: non-root UID 10001.
- Terminal screenshot showing Docker health `healthy` and the `/health` response.
- Include the short explanation above, using your own words.

No HTML evidence screenshots are generated for this task. Actual verification
outputs are saved under `outputs/task6/6_2/`.

## Verified result

The built image starts successfully with champion version 2. Its user is
`uid=10001(visionguard)`, Docker reports `healthy`, debug is off, and a real JPEG
upload returns a saved prediction visible in SQLite-backed history. The runtime
contains `tensorflow-cpu==2.21.0` and no NVIDIA, MLflow or DVC packages.
The test container is left running at http://127.0.0.1:8082.
