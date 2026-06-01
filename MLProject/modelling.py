#%%
import mlflow
import mlflow.sklearn

import pandas as pd

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

#%%
# load dataset
df = pd.read_csv(
    'mental_preprocessing/mental_clean.csv'
)

df.head()

#%%
# split feature target
X = df.drop(
    'mental_health_risk',
    axis=1
)

y = df[
    'mental_health_risk'
]

# %%
# split train test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#%%
mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# PREDICTION
y_pred = model.predict(
    X_test
)

# PROBABILITY PREDICTION
y_prob = model.predict_proba(
    X_test
)

# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class='ovr'
)

# ==================================================
# ADDITIONAL EVALUATION
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred
)

# ==================================================
# PRINT RESULT
# ==================================================

print("=" * 50)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

print("=" * 50)

