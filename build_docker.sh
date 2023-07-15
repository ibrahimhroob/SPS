#!/usr/bin/env bash

image_name=mos4d

docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${image_name} $(dirname "$0")/

