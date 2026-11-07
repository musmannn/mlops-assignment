#!/usr/bin/env bash
# Task 6.1 - Docker warm-up with a ready-made image (nginx). Screenshot each numbered step.
set -x
mkdir -p ~/docker-warmup-24L-0672 && echo "<h1>24L-0672 Docker warm-up</h1>" > ~/docker-warmup-24L-0672/index.html

# 1. run a container from a ready-made image, detached, with a name
docker run -d --name warmup-nginx \
  -p 8080:80 \
  -v ~/docker-warmup-24L-0672:/usr/share/nginx/html:ro \
  nginx:1.27-alpine
# 2. port mapping: host 8080 -> container 80
curl -s http://localhost:8080
# 3. volume: edit the file on the host, the container serves the new version immediately
echo "<p>edited on the host at $(date)</p>" >> ~/docker-warmup-24L-0672/index.html
curl -s http://localhost:8080
# 4. logs
docker logs --tail 5 warmup-nginx
# 5. shell inside the container
docker exec warmup-nginx sh -c 'whoami; hostname; ls /usr/share/nginx/html; nginx -v'
# (interactive version: docker exec -it warmup-nginx sh)
# 6. list containers and images
docker ps -a --filter name=warmup
docker images nginx
# 7. remove the container and the image
docker stop warmup-nginx && docker rm warmup-nginx
docker rmi nginx:1.27-alpine
docker ps -a --filter name=warmup
