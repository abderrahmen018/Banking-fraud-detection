import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# === Chargement des données ===
df = pd.read_csv("asset/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalisation
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Chargement des modèles
models_loaders = {
    "random_forest": load("models/random_forest_model.joblib"),
    "logistic_regression": load("models/logistic_regression_model.joblib"),
    "xgboost": load("models/xgboost_model.joblib"),
    "sgd_classifier": load("models/sgd_classifier_model.joblib"),
    "decision_tree": load("models/decision_tree_model.joblib")
}

# === Fonction pour afficher une image dans Tkinter ===
def show_image(path, label_widget):
    image = Image.open(path)
    image = image.resize((300, 250))  # Redimensionner l'image
    photo = ImageTk.PhotoImage(image)
    label_widget.image = photo
    label_widget.config(image=photo)

# === Fonction principale de prédiction ===
def predict_model():
    model_name = model_var.get()
    if not model_name:
        messagebox.showwarning("Erreur", "Veuillez sélectionner un modèle.")
        return

    model = models_loaders[model_name]
    y_pred = model.predict(X_test)

    # Rapport classification
    report = classification_report(y_test, y_pred, target_names=["Non-fraude", "Fraude"])
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"➡️ Modèle sélectionné : {model_name}\n\n")
    output_text.insert(tk.END, report)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Non-fraude", "Fraude"],
                yticklabels=["Non-fraude", "Fraude"])
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig("asset/conf_matrix.png")
    plt.close()

    # Courbe ROC-AUC
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel("Faux Positifs (FPR)")
        plt.ylabel("Vrais Positifs (TPR)")
        plt.title("Courbe ROC")
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()
        plt.savefig("asset/roc_curve.png")
        plt.close()
    except AttributeError:
        if os.path.exists("asset/roc_curve.png"):
            os.remove("asset/roc_curve.png")

    # Affichage des images
    show_image("asset/conf_matrix.png", image_label1)
    if os.path.exists("asset/roc_curve.png"):
        show_image("asset/roc_curve.png", image_label2)
    else:
        image_label2.config(image="")

# === Interface Toplevel ===
def open_prediction_window(parent):
    global model_var, output_text, image_label1, image_label2

    csv_root = tk.Toplevel(parent)
    csv_root.title("Detection de Fraude - Selection du Modele")
    csv_root.geometry("750x850")

    model_var = tk.StringVar()
    model_label = tk.Label(csv_root, text="Selectionner le modele :", font=("Arial", 12))
    model_label.pack(pady=10)

    model_dropdown = ttk.Combobox(csv_root, textvariable=model_var, state="readonly", font=("Arial", 11))
    model_dropdown['values'] = list(models_loaders.keys())
    model_dropdown.pack(pady=5)

    predict_button = tk.Button(csv_root, text="Predict", command=predict_model, font=("Arial", 12), bg="#4CAF50", fg="white")
    predict_button.pack(pady=20)

    output_text = tk.Text(csv_root, height=15, width=85, font=("Terminal", 15))
    output_text.pack(padx=10, pady=10)

    # Zone pour les images
    image_frame = tk.Frame(csv_root)
    image_frame.pack(pady=10)

    image_label1 = tk.Label(image_frame)
    image_label1.pack(side=tk.LEFT, padx=10)

    image_label2 = tk.Label(image_frame)
    image_label2.pack(side=tk.LEFT, padx=10)

    # Bouton retour
    btn_retour = tk.Button(csv_root, text="⬅️ Retour", command=csv_root.destroy)
    btn_retour.pack(pady=10)
