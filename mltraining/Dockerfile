FROM python:3.10

WORKDIR /app

COPY train.py .
COPY student_performance.csv .

RUN pip install --no-cache-dir scikit-learn pandas mlflow

CMD python train.py

     