from flask import Flask, request, render_template
import mlflow
import pymongo
import os
import numpy as np
from mlflow.pyfunc import load_model

app=Flask(__name__)

# MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://mongo:27017/")  # mongo = service name
db = mongo_client["student_db"]
collection = db["predictions"]

#mlflow connection
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_server:5000"))

model_name = "student-pass-model"
model_uri = f"models:/{model_name}/latest"
model = load_model(model_uri)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form["study_hours"])
    sleep_hours = float(request.form["sleep_hours"])

    prediction=model.predict(np.array([[study_hours, sleep_hours]]))[0]

    collection.insert_one({
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "predicted_pass": int(prediction)
    })

    return render_template("index.html", prediction=int(prediction))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
