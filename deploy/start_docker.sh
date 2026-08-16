#!/bin/bash
set -e

REGION="eu-north-1"
ACCOUNT_ID="512902042984"
REPOSITORY="hybrid-music-recommender-system"
IMAGE_TAG="latest"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker pull $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY:$IMAGE_TAG

if [ "$(docker ps -aq -f name=hybrid-rexus)" ]; then
    docker stop hybrid-rexus
    docker rm hybrid-rexus
fi

docker run -d --name hybrid-rexus -p 8000:8000 $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY:$IMAGE_TAG