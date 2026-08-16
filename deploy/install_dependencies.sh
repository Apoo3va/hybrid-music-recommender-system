#!/bin/bash
set -e

apt-get update
apt-get install -y docker.io unzip

systemctl start docker
systemctl enable docker

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip
./aws/install --update
rm -rf awscliv2.zip aws

usermod -aG docker ubuntu