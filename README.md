# Automated Continuous Integration & MLOps Orchestration Pipeline

## Description
The core DevOps and automation backbone for serving and monitoring the Telco Churn prediction system. Handles automated containerization and cloud metric tracking.

## CI/CD Workflow
The GitHub Actions pipeline (`.github/workflows/ci_retrain.yml`) headlessly pulls the latest champion model from the DagsHub MLflow Registry, packages it into a production-grade Docker container using `mlflow models build-docker`, and pushes the image directly to Docker Hub at `lukmanhakim404/credit_scoring_model:latest`.

## Monitoring & Alerting Stack
Architectural overview of the automated monitoring framework using Prometheus to scrape the FastAPI `/metrics` endpoint, coupled with a Grafana instance reading metrics from the user target workspace dashboard (`lukman404`). Includes real-time alerting limits configured for system operational flags.
