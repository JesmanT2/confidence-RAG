# AWS Deployment (ECR + EC2)

## Prerequisites

- AWS CLI configured with IAM permissions for ECR and EC2
- Docker installed on local machine and EC2
- Environment variables:
  - `AWS_REGION`
  - `AWS_ACCOUNT_ID`
  - `REPO_NAME`
  - Optional: `IMAGE_TAG`

## Build and push to ECR

```bash
chmod +x deploy/aws/deploy_to_ecr_ec2.sh
AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 REPO_NAME=confidence-rag IMAGE_TAG=v1 ./deploy/aws/deploy_to_ecr_ec2.sh
```

## Run on EC2

```bash
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/confidence-rag:v1
docker run -d --name confidence-rag -p 8000:8000 123456789012.dkr.ecr.us-east-1.amazonaws.com/confidence-rag:v1
```

## Health check

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
```
