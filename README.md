**Credit Card Fraud Detection**

This project implements a sequence based LSTM (Long Short Term Memory) to detect credit card fraud. The model analyzes the sequence of transactions over a sliding window in order to detect fraud, is trained on the Kaggle Credit Card Fraud dataset and uses SMOTE (Synthetic Minority Oversampling Technique) and scaling to balance data.
The model is also evaluated based on how it does on: Precision, Recall, F1-Score, and AUC-ROC

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
The dataset has 284,807 credit card transactions, of which only 492 are fraudulent. Features V1–V28 were transformed via PCA transformation for confidentiality, with the only untransformed features being Time and Amount. The target column is Class, where 1 is fraud and 0 is not fraud 

Requirements
- python 3.7+
- numpy
- pandas
- scikit-learn
- imbalanced-learn

Install dependencies with:
pip install numpy pandas scikit-learn imbalanced-learn

(We also have a requirements.txt)

How To Run:
- Clone or download this repo
- Go to //www.kaggle.com/datasets/mlg-ulb/creditcardfraud (make sure to be logged in) and download dataset
- Place file in the root of the project directory, file structure should look like:
  your-project-folder/
      main.py
      model.py
      ...
      creditcard.csv
- Run program: python main.py creditcard.csv

Ouput:
After training, the program prints a loss trable across all epochs and an evaluation metrics table on the test dataset
