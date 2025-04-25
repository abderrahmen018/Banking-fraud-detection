# Simple script To train Models

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression , SGDClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# === Étape 1 : Charger les données ===
df = pd.read_csv("asset/creditcard.csv")
print("Distribution des classes initiale :\n", df["Class"].value_counts())

# === Étape 2 : Séparer X et y ===
X = df.drop("Class", axis=1)
y = df["Class"]

# === Étape 3 : Normaliser la colonne 'Amount' ===
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# === Étape 4 : Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Étape 5 : SMOTE pour équilibrer les données d'entraînement ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nDonnées après SMOTE :")
print(pd.Series(y_train_resampled).value_counts())


classifiers = {
    "random_forest" : RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression" : LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "xgboost" : XGBClassifier(eval_metric='logloss', random_state=42),
    "svc" : SVC(probability=True),
    "sgd_classifier" : SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
    "decision_tree" : DecisionTreeClassifier()
}

# === Étape 6 : Entraînement Du models + Sauvegarde ===

def train_model(model_name , classifiers , X_train_resampled , y_train_resampled ):
    # Choix du Model
    print(f"➡️ Entrenement du Model {model_name}")
    model = classifiers[model_name]

    # Entrenement 
    model.fit(X_train_resampled, y_train_resampled)

    # Sauvegard du model
    dump(model, f"{model_name}_model.joblib")
    print(f"✅ sauvegardé dans {model_name}_model.joblib")

train_model("xgboost" , classifiers , X_train_resampled , y_train_resampled)