#!/bin/sh

docker build -t dog_breed_identifier_web .
docker image tag dog_breed_identifier_web anhvnn810/dog_breed_identifier_web
docker push anhvnn810/dog_breed_identifier_web