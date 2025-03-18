import pandas as pd

# Load predictions and true labels
predictions = pd.read_csv('C:/projects/Credit-Card-Fraud-Detection/data/processed/batch_predictions.csv')['Predictions']
y_test = pd.read_csv('C:/projects/Credit-Card-Fraud-Detection/data/processed/y_test.csv')['Class']

# Compute accuracy
accuracy = (predictions == y_test).mean()
print(f"Accuracy on test set: {accuracy:.4f}")

# Compute additional metrics (optional)
from sklearn.metrics import recall_score, precision_score, f1_score
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")