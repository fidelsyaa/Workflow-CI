#%%
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import shutil
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

# Clean up existing mlruns if corrupt
if os.path.exists("mlruns"):
    try:
        shutil.rmtree("mlruns")
        print("✅ Old mlruns removed")
    except:
        print("⚠️ Could not remove old mlruns")

# Set tracking URI ke local directory
mlflow.set_tracking_uri("file:./mlruns")

# Buat experiment baru
mlflow.set_experiment("mental_health_experiment")

#%%
# load dataset
df = pd.read_csv('mental_preprocessing/mental_clean.csv')

#%%
# split feature target
X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

#%%
# split train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%%
# TRAINING dengan MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # PREDICTION
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # PRINT RESULT
    print("=" * 50)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", roc_auc)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    print("=" * 50)
    
    print(f"\n✅ Run ID: {mlflow.active_run().info.run_id}")
    print(f"✅ Model saved at: mlruns/")