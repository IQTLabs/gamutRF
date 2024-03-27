#! /usr/bin/env bash

# Script to update GamutRF and Birdseye

echo "Pulling new GamutRF updates..."
git pull

echo "Pulling latest docker images..."
docker rmi $(docker images | grep 'iqtlabs')
docker compose -f orchestrator.yml -f torchserve.yml -f torchserve-cuda.yml -f torchserve.yml pull

if [ -d ~/BirdsEye ]; then
    cd ~/Birdseye
    git pull
    docker compose -f geolocate.yml pull
    cd -
fi