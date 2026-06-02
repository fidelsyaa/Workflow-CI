import mlflow
import mlflow.sklearn
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =========================
# FIX MLflow Tracking (WAJIB)
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# =========================
# Load dataset
# =========================
df = pd.read_csv('mental_preprocessing/mental_clean.csv')

X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MLflow Training
# =========================
run_name = f"run_{int(time.time())}"

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob, multi_class='ovr'))

    mlflow.sklearn.log_model(model, "model")

    print("Run ID:", mlflow.active_run().info.run_id)