# 🕵️ Fraud Detection System

This project is a machine learning application with a graphical interface for detecting fraudulent banking transactions.

---

## 🚀 Features

- 🔍 **Predict from a CSV file**
- 🧪 **Manually test a transaction**
- ⚖️ **Compare two machine learning models**
- User interface developed using **Tkinter**
- Machine learning models such as **Random Forest**, **XGBoost**, **Logistic Regression**, etc.

---

## 🖥️ Technologies

- Python 3
- Tkinter (User Interface)
- Scikit-learn, XGBoost
- Pandas, Numpy, Matplotlib, Seaborn
- joblib (for loading models)
- Git

## Setup Instructions

### 1. Clone the Repository

Start by cloning the repository to your local machine:

'''
git clone https://github.com/abderrahmen018/Banking-fraud-detection.git
'''

### 2. Create the `asset` Folder

Create a folder named `asset` in the root directory of the project. This folder will be used to store the following files:
- The `creditcard.csv` file (dataset used for predictions).
- The result images (graphs, confusion matrices, etc.) generated during the analysis.

### 3. Download the `creditcard.csv` File

The [`creditcard.csv`](https://www.kaggle.com/code/chanchal24/credit-card-fraud-detection) dataset is required to run the prediction and transaction testing functions. You can download the dataset from [this link](https://www.kaggle.com/mlg-ulb/creditcardfraud) or another source. Once downloaded, place the file inside the `asset` folder.

### 4. Install Dependencies

### 5. Running the Application

Once everything is set up, you can run the project by executing the main script:

'''
python fraud_app.py
'''

This will launch the graphical user interface (GUI) where you can:
- **Select a model** for predictions from a CSV file.
- **Test a single transaction** by entering transaction details manually.
- **Compare multiple models** to evaluate their performance.

### 6. Results

- **Prediction results** will be displayed in the GUI after a model is selected and predictions are made.
- **Graphs and matrices** (like confusion matrix and ROC curve) will be saved in the `asset` folder.
  - `asset/conf_matrix.png` (Confusion Matrix)
  - `asset/roc_curve.png` (ROC Curve)

## Important Notes

- Ensure that the `creditcard.csv` file is placed inside the `asset` folder before running the application.
- The models are already pre-trained and stored in the `models` folder. You can add or replace models by placing new `.joblib` files in that folder.
- You may need to adjust file paths if you change the folder structure or model names.

---

## License

This project is open source.
