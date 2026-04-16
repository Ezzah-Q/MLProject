import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Establish defs and labels
def evaluate_model(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    aucroc = roc_auc_score(y_true, y_prob)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1score)
    print("AUC-ROC: ", aucroc)
    
    return precision, recall, f1score, aucroc