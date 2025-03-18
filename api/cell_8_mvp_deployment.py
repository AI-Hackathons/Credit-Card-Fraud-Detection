import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for request validation
class PredictionRequest(BaseModel):
    features: list

class FraudPredictor:
    def __init__(self, rf_model_path, iso_forest_path, threshold=0.27):
        """
        Initialize the predictor with pre-trained models and threshold.
        
        Args:
            rf_model_path (str): Path to the Random Forest model
            iso_forest_path (str): Path to the Isolation Forest model
            threshold (float): Prediction threshold (0.27 from Cell 7)
        """
        try:
            self.rf_model = joblib.load(rf_model_path)
            self.iso_forest = joblib.load(iso_forest_path)
            self.threshold = threshold
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def predict_fraud(self, data):
        """
        Predict fraud for input data using the hybrid model.
        
        Args:
            data (pd.DataFrame or np.ndarray): Input features
        
        Returns:
            np.ndarray: Binary predictions (0: Non-Fraud, 1: Fraud)
        """
        try:
            # Define expected columns (31 features) in the order used during training
            expected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                              'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_log', 'Amount_squared']

            # Ensure data is in DataFrame format
            if isinstance(data, np.ndarray):
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)  # Convert 1D array to 2D
                if data.shape[1] != len(expected_columns):
                    raise ValueError(f"Expected {len(expected_columns)} features, got {data.shape[1]}")
                data = pd.DataFrame(data, columns=expected_columns)
            elif isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in expected_columns):
                    missing_columns = [col for col in expected_columns if col not in data.columns]
                    for col in missing_columns:
                        data[col] = 0.0  # Add missing columns with default value
            else:
                raise ValueError("Input must be a NumPy array or pandas DataFrame")

            # Generate predictions
            start_time = time.time()
            rf_proba = self.rf_model.predict_proba(data)[:, 1]
            rf_pred = (rf_proba >= self.threshold).astype(int)
            iso_pred = (self.iso_forest.predict(data) == -1).astype(int)
            weighted_pred = 0.9 * rf_pred + 0.1 * iso_pred
            predictions = (weighted_pred >= 0.5).astype(int)
            end_time = time.time()
            logger.info(f"Predictions generated in {end_time - start_time:.2f} seconds for {len(data)} samples.")
            return predictions
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

# Batch prediction function
def run_batch_prediction(predictor, test_data_path):
    """
    Run batch predictions on test data.
    
    Args:
        predictor (FraudPredictor): Initialized predictor object
        test_data_path (str): Path to test data CSV
    """
    try:
        # Load test data
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop(columns=['Class']) if 'Class' in test_df.columns else test_df
        
        # Predict
        predictions = predictor.predict_fraud(X_test)
        
        # Save predictions to a file
        pd.Series(predictions, name='Predictions').to_csv(
            'C:/projects/Credit-Card-Fraud-Detection/data/processed/batch_predictions.csv', index=False
        )
        logger.info("Batch predictions saved to batch_predictions.csv.")
        
        # Log and display results
        logger.info("Batch prediction completed.")
        print("Sample Predictions (first 10):", predictions[:10])
        if 'Class' in test_df.columns:
            accuracy = np.mean(predictions == test_df['Class'])
            logger.info(f"Accuracy on test set: {accuracy:.4f}")
            print(f"Accuracy on test set: {accuracy:.4f}")
        else:
            logger.info("Class column not found in test data. Load y_test.csv to compute accuracy.")
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise

# FastAPI setup
app = FastAPI(title="Credit Card Fraud Detection API", description="API for real-time credit card fraud prediction.")

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    API endpoint for real-time fraud prediction.
    Expects JSON with feature array.
    """
    try:
        features = np.array(request.features)
        predictor = FraudPredictor(
            'C:/projects/Credit-Card-Fraud-Detection/models/optimized_rf_model.pkl',
            'C:/projects/Credit-Card-Fraud-Detection/models/optimized_iso_forest.pkl'
        )
        predictions = predictor.predict_fraud(features)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = FraudPredictor(
            'C:/projects/Credit-Card-Fraud-Detection/models/optimized_rf_model.pkl',
            'C:/projects/Credit-Card-Fraud-Detection/models/optimized_iso_forest.pkl'
        )
        
        # Run batch prediction on test data
        run_batch_prediction(predictor, 'C:/projects/Credit-Card-Fraud-Detection/data/processed/X_test.csv')
        
        # Start FastAPI server
        logger.info("Starting FastAPI server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Main execution error: {e}")