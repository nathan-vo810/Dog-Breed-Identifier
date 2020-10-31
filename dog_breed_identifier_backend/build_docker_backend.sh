#!/bin/sh

docker build -t dog_breed_identifier_backend .
docker image tag dog_breed_identifier_backend anhvnn810/dog_breed_identifier_backend:pytorch
docker push anhvnn810/dog_breed_identifier_backend:pytorch