import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Charger les modèles
models = {
    "random_forest": load("models/random_forest_model.joblib"),
    "logistic_regression": load("models/logistic_regression_model.joblib"),
    "xgboost": load("models/xgboost_model.joblib"),
    "sgd_classifier": load("models/sgd_classifier_model.joblib"),
    "decision_tree": load("models/decision_tree_model.joblib")
}

def open_transaction_test_window(parent):
    test_win = tk.Toplevel(parent)
    test_win.title("Tester une transaction")
    test_win.geometry("800x700")

    model_var = tk.StringVar()
    entries = {}

    tk.Label(test_win, text="Selection du modele :", font=("Arial", 12)).pack(pady=10)
    model_menu = ttk.Combobox(test_win, textvariable=model_var, values=list(models.keys()), state="readonly")
    model_menu.pack()

    # === Zone de saisie ===
    form_frame = tk.Frame(test_win)
    form_frame.pack(pady=20)

    # Liste des colonnes
    columns = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]

    # Répartition des colonnes en 4 colonnes visuelles
    num_columns = 4

    for i, col in enumerate(columns):
        row = i // num_columns
        col_pos = i % num_columns

        tk.Label(form_frame, text=col).grid(row=row, column=col_pos*2, padx=5, pady=5, sticky="e")
        entry = tk.Entry(form_frame, width=20)
        entry.grid(row=row, column=col_pos*2 + 1, padx=5, pady=5)
        entries[col] = entry

    # === Résultat ===
    result_label = tk.Label(test_win, text="", font=("Arial", 14, "bold"))
    result_label.pack(pady=10)

    # === Fonction de prédiction ===
    def predict_transaction():
        model_name = model_var.get()
        if not model_name:
            messagebox.showerror("Erreur", "Selectionner un modele.")
            return

        try:
            input_data = []
            for col in columns:
                val = float(entries[col].get())
                input_data.append(val)

            # Normaliser uniquement le montant
            amount_index = columns.index("Amount")
            scaler = StandardScaler()
            amount_scaled = scaler.fit_transform(np.array(input_data[amount_index]).reshape(-1, 1))[0][0]
            input_data[amount_index] = amount_scaled

            input_array = np.array(input_data).reshape(1, -1)

            model = models[model_name]
            prediction = model.predict(input_array)[0]

            if prediction == 0:
                result = "🟢 Non-Fraude"
            else:
                result = "🔴 Fraude detecte"
            result_label.config(text=f"Resultat : {result}", fg="green" if prediction == 0 else "red")

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs avec des valeurs numériques.")

    # === Bouton prédire ===
    predict_btn = tk.Button(test_win, text="Predire", command=predict_transaction, font=("Arial", 12), bg="#007ACC", fg="white")
    predict_btn.pack(pady=20)

    # === Bouton retour ===
    back_btn = tk.Button(test_win, text="⬅ Retour", command=test_win.destroy)
    back_btn.pack(pady=10)
