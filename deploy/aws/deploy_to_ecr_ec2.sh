#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 REPO_NAME=confidence-rag IMAGE_TAG=v1 ./deploy/aws/deploy_to_ecr_ec2.sh

: "${AWS_REGION:?Set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${REPO_NAME:?Set REPO_NAME}"
: "${IMAGE_TAG:=latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t "${REPO_NAME}:${IMAGE_TAG}" .
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

echo "Pushed image: ${ECR_URI}"
echo "Run on EC2:"
echo "docker run -d --name confidence-rag -p 8000:8000 ${ECR_URI}"
