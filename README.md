Credit Card Fraud Detection

This project implements a sequence based LSTM to detect credit card fraud. The model analyzes the sequence of transactions in order to detect fraud, is trained on the Kaggle Credit Card Fraud dataset and uses SMOTE and scaling to balance data.
The model is also evaluated based on how it does on: Precision, Recall, F1-Score, and AUC-ROC

How To Run:
- Install Python & Python Packages: pip install numpy pandas scikit-learn imbalanced-learn
- Download the Kaggle Dataset: creditcard.csv 
- Place file in project folder
- Run program: python main.py creditcard.csv