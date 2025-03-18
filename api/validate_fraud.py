import pandas as pd

# Load predictions and true labels
predictions = pd.read_csv('C:/projects/Credit-Card-Fraud-Detection/data/processed/batch_predictions.csv')['Predictions']
y_test = pd.read_csv('C:/projects/Credit-Card-Fraud-Detection/data/processed/y_test.csv')['Class']

# Find fraud predictions and their true labels
fraud_indices = predictions[predictions == 1].index
fraud_results = pd.DataFrame({'True_Label': y_test[fraud_indices], 'Predicted_Label': predictions[fraud_indices]})
print("Fraud predictions and true labels:")
print(fraud_results.head(10))  # First 10 fraud predictions