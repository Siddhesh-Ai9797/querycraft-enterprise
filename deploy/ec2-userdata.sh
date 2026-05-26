#!/bin/bash
set -e
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 753093280070.dkr.ecr.us-east-1.amazonaws.com

mkdir -p /app
cat > /app/docker-compose.yml << EOF
version: "3.8"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres123
      POSTGRES_DB: chinook
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  app:
    image: 753093280070.dkr.ecr.us-east-1.amazonaws.com/querycraft-enterprise:latest
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres123@postgres:5432/chinook
      OPENAI_API_KEY: YOUR_OPENAI_KEY_HERE
      INFERENCE_BACKEND: openai
    depends_on:
      - postgres

volumes:
  pgdata:
EOF

cd /app
docker-compose up -d