# File: mlflow_server/Dockerfile
FROM python:3.10
WORKDIR /mlflow

# Install MLflow and dependencies
RUN pip install mlflow==2.14 psycopg2-binary boto3

# Expose port 5000 (MLflow UI)
EXPOSE 5000

# Copy entrypoint script: Copy the file entrypoint.sh from your computer → inside the Docker image.
COPY entrypoint.sh .

# Make script executable: chmod +x = give permission to run the file like a program.
RUN chmod +x entrypoint.sh

# Default command: "When this container starts, run entrypoint.sh"
CMD ["./entrypoint.sh"] 