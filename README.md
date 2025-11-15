Trainer container → trains Random Forest → logs parameters & metrics to MLflow
MLflow container → backend-store = SQLite, artifacts = volume
Flask container → fetches latest model from MLflow, makes predictions
MongoDB container → stores inputs + predictions
