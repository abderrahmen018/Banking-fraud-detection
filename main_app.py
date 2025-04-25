import tkinter as tk
from tkinter import ttk

def open_csv_prediction():
    import prediction_csv
    prediction_csv.open_prediction_window(root)

def open_transaction_test():
    import test_transaction
    test_transaction.open_transaction_test_window(root)

def open_model_comparison():
    import compare_models
    compare_models.open_model_comparison_window(root)

# === Fenêtre principale ===
root = tk.Tk()
root.title("Detection Fraude bancaire")
root.geometry("400x300")
root.resizable(False, False)

title = tk.Label(root, text="Marhba 👋", font=("Arial", 18, "bold"))
title.pack(pady=20)

subtitle = tk.Label(root, text="Choisissez une fonctionnalité :", font=("Arial", 12))
subtitle.pack(pady=10)

btn_csv = tk.Button(root, text="Predire depuis le fichier CSV", command=open_csv_prediction,
                    font=("Arial", 11), width=30, bg="#007ACC", fg="white")
btn_csv.pack(pady=10)

btn_test = tk.Button(root, text="Tester Transaction", command=open_transaction_test,
                     font=("Arial", 11), width=30, bg="#4CAF50", fg="white")
btn_test.pack(pady=10)

btn_compare = tk.Button(root, text="Comparer Deux Modeles", command=open_model_comparison,
                        font=("Arial", 11), width=30, bg="#F57C00", fg="white")
btn_compare.pack(pady=10)

root.mainloop()
