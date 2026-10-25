# Task 6.5 — operations and troubleshooting

Run these commands in order from the assignment folder. They affect only this
Compose project. Do not use `down -v`; the database volume should be retained.

## 1. Stop

```bash
docker compose stop
docker compose ps -a
```

Screenshot: both services stopped.

## 2. Start

```bash
docker compose start --wait
docker compose ps
```

Screenshot: both services healthy.

## 3. Restart

```bash
docker compose restart
docker compose up -d --wait
docker compose ps
```

The second command waits for readiness after the restart.

## 4. Read each service's logs

```bash
docker compose logs --tail 15 web
docker compose logs --tail 15 db
```

Screenshot: output from both services.

## 5. Open a shell in web

```bash
docker compose exec web sh
```

Inside the container:

```sh
id
pwd
exit
```

## 6. Open a shell in db

```bash
docker compose exec db sh
```

Inside the container:

```sh
id
pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
exit
```

Screenshot the commands inside each shell. The recorded automated run used `-T`
and piped shell input; these manual commands provide an interactive terminal.

## 7. Trigger a missing-model failure

The explicit override sets `MODEL_DIR=/models/missing`. It does not rename or
remove the real model files and is not loaded during normal Compose startup.

```bash
docker compose -f compose.yaml -f compose.failure.yaml up -d --no-deps --force-recreate web
docker wait "$(docker compose ps -a -q web)"
docker compose logs --tail 60 web
```

Screenshot the log line containing:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/models/missing/metadata.json'
```

Diagnosis from logs: the web worker cannot find the model metadata in its
configured model directory.

## 8. Fix and verify

```bash
docker compose up -d --no-deps --force-recreate --wait web
docker compose ps
curl http://127.0.0.1:8083/health
```

Screenshot: healthy services and successful health response.

The normal Compose file restores `MODEL_DIR=/models/champion`. Recreating the
container applies this environment change; `restart` alone does not apply updated
configuration. [Docker restart documentation](https://docs.docker.com/reference/cli/docker/compose/restart/).

## Short report explanation

The logs showed `FileNotFoundError` for `/models/missing/metadata.json`, identifying
an incorrect model directory. Recreating web with the normal Compose settings
restored `/models/champion` and a healthy service.

## Saved evidence

Actual outputs for each operation, both shells, failure logs, diagnosis and
recovery are saved under `outputs/task6/6_5/`. No HTML evidence images are used.
