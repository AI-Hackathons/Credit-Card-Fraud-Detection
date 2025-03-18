# Credit Card Fraud Detection Project

## Phase 1: Model Development and MVP Deployment (Cells 6-8)

### Overview
- **Dataset**: UCI Credit Card Fraud Detection dataset (anonymized with PCA features).
- **Features**: V1 to V28 (PCA components), Amount, Amount_log, Amount_squared (31 features total).
- **Test Set Size**: 57,149 samples (56,848 non-fraud, 301 fraud).
- **Targets**:
  - Recall >0.70
  - Precision >0.70
  - F1-score >0.70
  - ROC-AUC >0.80–0.90

### Cell 6: Hybrid Model Development
- Built a hybrid model combining Random Forest (RF) and Isolation Forest (IF).
- Initial metrics at threshold 0.27:
  - Recall: 0.9136
  - Precision: 0.9483
  - F1-score: 0.9306
  - ROC-AUC: 0.9841

### Cell 7: Model Tuning
- Tuned RF (`max_depth=15`, `n_estimators=30`) and IF (`contamination=0.005`).
- Evaluated thresholds; best at 0.27 (same metrics as Cell 6):
  - Recall: 0.9136
  - Precision: 0.9483
  - F1-score: 0.9306
  - ROC-AUC: 0.9841
- Note: Perfect test set accuracy (1.00) suggests potential overfitting.

### Cell 8: FastAPI MVP Deployment
- Deployed the hybrid model using FastAPI with `/predict` endpoint and batch prediction.
- **API Tests**:
  - Request 1: {"features": [[-1.3598071336738, ..., 19071.74, 9.856015369447933, 363731266.6276001]]}, Response: {"predictions": [0]}
  - Request 2: {"features": [[-0.4311030103069341, ..., 17424.19, 9.765672139443025, 303602397.1561]]}, Response: {"predictions": [0]}
  - Request 3: {"features": [[-2.31222654232647, ..., 1.00, 0.0, 1.0]]}, Response: {"predictions": [1]}
- **Batch Prediction**:
  - Sample Predictions (first 10): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  - Execution Time: 0.61 seconds for 57,149 samples
  - Accuracy: 0.9993
  - Recall: 0.9136
  - Precision: 0.9483
  - F1-score: 0.9306
- **Fraud Prediction Validation**:
  - Sampled first 10 predicted fraud cases: 9/10 were correct (True_Label=1), 1 false positive (True_Label=0).
  - Aligns with precision (0.9483); ~15 false positives estimated across all predictions.
  - Estimated ~26 fraud cases missed (false negatives), consistent with recall (0.9136).
- **Validation Summary**: Predictions validated against y_test.csv; metrics match Cell 7. API tests and batch results align with ground truth, confirming correctness within the dataset.
- **Note**: Near-perfect accuracy (0.9993) suggests overfitting to the UCI dataset; Phase 2 will validate with real-world data.

## Next Steps (Phase 2)
- Source a real-world dataset (2023–2025) with raw features.
- Engineer interpretable features (e.g., transaction frequency, time-based patterns).
- Validate and retrain the model to address overfitting.
- Enhance the API with authentication, input validation, and batch prediction endpoint.