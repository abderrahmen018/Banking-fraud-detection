import tkinter as tk
from tkinter import ttk, messagebox
from joblib import load
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# === Chargement des données ===
df = pd.read_csv("asset/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalisation uniquement de "Amount"
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Chargement des modèles ===
models = {
    "random_forest": load("models/random_forest_model.joblib"),
    "logistic_regression": load("models/logistic_regression_model.joblib"),
    "xgboost": load("models/xgboost_model.joblib"),
    "sgd_classifier": load("models/sgd_classifier_model.joblib"),
    "decision_tree": load("models/decision_tree_model.joblib")
}


def open_model_comparison_window(parent):
    compare_win = tk.Toplevel(parent)
    compare_win.title("Comparer deux modèles")
    compare_win.geometry("850x700")

    tk.Label(compare_win, text="Sélectionnez deux modèles à comparer :", font=("Arial", 12)).pack(pady=10)

    model_var1 = tk.StringVar()
    model_var2 = tk.StringVar()

    frame = tk.Frame(compare_win)
    frame.pack(pady=5)

    ttk.Combobox(frame, textvariable=model_var1, values=list(models.keys()), state="readonly", width=25).grid(row=0, column=0, padx=10)
    ttk.Combobox(frame, textvariable=model_var2, values=list(models.keys()), state="readonly", width=25).grid(row=0, column=1, padx=10)

    output_text = tk.Text(compare_win, height=25, width=100, font=("Terminal", 12))
    output_text.pack(pady=20)

    result_label = tk.Label(compare_win, text="", font=("Arial", 14, "bold"))
    result_label.pack(pady=10)

    def compare_models():
        m1 = model_var1.get()
        m2 = model_var2.get()

        if not m1 or not m2:
            messagebox.showerror("Error", "Selectionner deux modeles.")
            return

        if m1 == m2:
            messagebox.showwarning("Warning", "les modeles dois Etre differents.")
            return

        model1 = models[m1]
        model2 = models[m2]

        y_pred1 = model1.predict(X_test)
        y_pred2 = model2.predict(X_test)

        f1_m1 = f1_score(y_test, y_pred1)
        f1_m2 = f1_score(y_test, y_pred2)

        output_text.delete("1.0", tk.END)

        output_text.insert(tk.END, f"====== {m1} ======\n")
        output_text.insert(tk.END, classification_report(y_test, y_pred1, target_names=["Non-Fraude", "Fraude"]))
        output_text.insert(tk.END, f"\n F1-Score : {f1_m1:.4f} \n\n\n")

        output_text.insert(tk.END, f"====== {m2} ======\n")
        output_text.insert(tk.END, classification_report(y_test, y_pred2, target_names=["Non-Fraude", "Fraude"]))
        output_text.insert(tk.END, f"\n F1-Score : {f1_m2:.4f} \n\n\n")

        diff = abs(f1_m1 - f1_m2) #Bech le resultat ykon positif
        if f1_m1 > f1_m2:
            better = m1
        else:
            better = m2
        
        #output_text.insert(tk.END, f"✅ Le modele performant en F1-Score est : {better} (== difference de {diff} ==)")
        
        result_label.config(text=f"✅ Le modele performant en F1-Score est : {better} (== difference de {diff:.4f} ==)", fg="green")


    compare_btn = tk.Button(compare_win, text="Comparer", command=compare_models, font=("Arial", 12),
                            bg="#F57C00", fg="white")
    compare_btn.pack(pady=10)

    back_btn = tk.Button(compare_win, text="⬅ Retour", command=compare_win.destroy)
    back_btn.pack(pady=10)
