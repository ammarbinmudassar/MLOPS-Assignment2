# Task 6.1 — Docker warm-up

## Short definitions

- **Image:** a packaged template containing an application and its dependencies.
- **Container:** an isolated instance created from an image, with its own writable state.
- **Layer:** a reusable, read-only filesystem change that forms part of an image.
- **Volume:** Docker-managed storage that can survive deleting a container.
- **Port mapping:** connecting a host port to a container port, such as `8081:80`.

References: [containers](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/),
[volumes](https://docs.docker.com/engine/storage/volumes/) and
[port publishing](https://docs.docker.com/engine/network/port-publishing/).

## Commands to understand or repeat

Run these in order. The completed demonstration removed its test containers and
Nginx image; the named volume remains. Names are specific to this lab.

```bash
docker pull nginx:alpine
docker volume create --label coursework=visionguard-6.1 visionguard-61-data
docker run -d --name visionguard-61-web --label coursework=visionguard-6.1 -p 127.0.0.1:8081:80 --mount source=visionguard-61-data,target=/usr/share/nginx/html nginx:alpine
docker port visionguard-61-web
curl -I http://127.0.0.1:8081/
docker inspect visionguard-61-web --format '{{json .Mounts}}'
docker exec visionguard-61-web sh -c 'echo "Task 6.1 volume survives container removal" > /usr/share/nginx/html/persistence.txt'
docker logs --tail 12 visionguard-61-web
```

`-d` runs in the background. The mapping sends localhost port 8081 to Nginx port
80. The named volume is mounted at Nginx's document folder.

Open a shell manually:

```bash
docker exec -it visionguard-61-web sh
whoami
pwd
cat /etc/alpine-release
ls /usr/share/nginx/html
exit
```

The recorded automated demonstration opened `sh` with `-i` and supplied these
commands through standard input; the manual command adds a terminal with `-t`.

List and remove the lab resources:

```bash
docker ps -a --filter label=coursework=visionguard-6.1
docker image ls nginx:alpine
docker stop visionguard-61-web
docker rm visionguard-61-web
docker ps -a --filter label=coursework=visionguard-6.1
docker run --rm --name visionguard-61-read --mount source=visionguard-61-data,target=/data,readonly nginx:alpine cat /data/persistence.txt
docker image rm nginx:alpine
docker image ls nginx:alpine
```

The second container reads the saved file after the first is deleted. This
confirms persistence. Removing an image is separate from removing a container.

## Evidence

`outputs/task6/6_1/console.txt` and `commands.json` contain the actual command
outputs, timestamps and exit codes. Screenshots in `screenshots/` display these
recorded outputs in a browser evidence sheet, rather than a live terminal.

| Screenshot | Demonstrates |
| --- | --- |
| `01_image.jpg` | Ready-made image and its digest |
| `02_volume.jpg` | Named volume creation |
| `03_run.jpg` | Container creation |
| `04_port.jpg` | Port mapping and HTTP response |
| `05_mount.jpg` | Mounted volume and saved file |
| `06_logs.jpg` | Container logs |
| `07_shell.jpg` | Commands inside the container shell |
| `08_list.jpg` | Container and image listing |
| `09_remove_container.jpg` | Stop, removal and empty listing |
| `10_persistence.jpg` | File surviving container removal |
| `11_remove_image.jpg` | Image removal and empty listing |

If the instructor requires terminal screenshots specifically, repeat the commands
above and capture the terminal. Keep the five definitions in your own words.
