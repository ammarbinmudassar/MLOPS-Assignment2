# Task 6.3 — Compose stack

## Short report explanation

Compose runs the web app and PostgreSQL as separate services. The web app reaches
`db:5432` through the Compose network and starts after the database health check
passes. PostgreSQL uses a named volume and publishes no host port.

Passwords come from the ignored `.env`; `.env.example` contains no password.
The champion model is mounted read-only, allowing model changes without rebuilding
the application image.

References: [startup readiness](https://docs.docker.com/compose/how-tos/startup-order/)
and [service-name networking](https://docs.docker.com/compose/how-tos/networking/).

## Run and capture terminal evidence

Private Compose settings are already prepared in `.env`. Do not screenshot its
contents or run `docker compose config` without `--quiet`, because resolved
configuration contains the database password.

```bash
docker compose config --quiet
docker compose up -d --build --wait
```

The verification stack is left running, so rebuilding is optional for screenshots.

```bash
docker compose ps
```

Both services should show `healthy`. Only web should have a host mapping:
`127.0.0.1:8083->8000/tcp`. The database may show `5432/tcp`, which is container
metadata, not a published host port.

```bash
docker inspect "$(docker compose ps -q db)" --format '{{json .HostConfig.PortBindings}}'
```

Expected: `{}` or `null`, proving no database host port binding.

```bash
docker compose exec web python -c "import socket; print('db resolves to:', socket.gethostbyname('db'))"
curl http://127.0.0.1:8083/health
```

The response should show `history_backend: postgresql` and `database_ready: true`.
Open http://127.0.0.1:8083 to use the Compose app.

```bash
docker volume ls --filter label=com.docker.compose.project=visionguard-assignment2
```

## Evidence to submit

- Repository link to `compose.yaml`.
- Terminal screenshot of `docker compose ps`: both services healthy.
- Terminal screenshot of database `PortBindings`: no published port.
- Short explanation above, including the read-only model mount choice.

PostgreSQL history is separate from the earlier local SQLite history. The existing
SQLite file is preserved. `docker compose down` retains the named volume; avoid
`down -v` when you want to keep saved history.

## Verified result

Both services are healthy. A real JPEG prediction was saved in PostgreSQL and
remained accessible after `down` followed by `up` without deleting volumes.
All 29 automated tests passed, including local SQLite compatibility and reserved
characters in database credentials. Outputs are in `outputs/task6/6_3/`.

A separate listener already uses localhost port 5432 on this machine. A global
TCP port probe therefore cannot demonstrate this container's publication state;
the database container's empty `HostConfig.PortBindings` is the relevant proof.
