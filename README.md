# Diabetes-Prediction
Prediting diabetes using Deep Neural Network with XGBoost

Methods used in the code are:

1. Data Preprocessing
   - Loaded CSV data using pandas
   - Split features (X) and target (y)
   - Standardized features using StandardScaler

2. Deep Neural Network (DNN)
   - Built a feedforward neural network using TensorFlow/Keras
   - Used Keras Tuner to perform hyperparameter tuning:
     - Number of layers
     - Units per layer
     - Dropout rate
     - Learning rate
   - Evaluated DNN model using Stratified K-Fold Cross-Validation
   - Performance metrics computed:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC AUC
       
3. XGBoost Classifier
   - Used xgboost.XGBClassifier for classification
   - Evaluated using Stratified K-Fold Cross-Validation
   - Computed same performance metrics as DNN
   - Extracted and plotted feature importances 
