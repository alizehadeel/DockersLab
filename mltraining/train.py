import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import mlflow
import os

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))#defined in the dockerfile

data=pd.read_csv('student_performance.csv')
X=data[["study_hours","sleep_hours"]]
Y=data["passed"]

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

mlflow.set_experiment("student-pass-experiment")

with mlflow.start_run():
    model=RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "student-pass-model") 

    print(f"Model trained with accuracy: {acc}")   