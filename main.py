# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt

# Load dataset
data = pd.read_csv('C:/Users/DCL/Desktop/Research Paper/diabetes.csv')
X = data.drop('Diabetes', axis=1)
y = data['Diabetes']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.array(X_scaled)
y = np.array(y)

# Keras model builder for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_scaled.shape[1],)))
    for i in range(hp.Int('layers', 1, 3)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', 32, 128, step=32), activation='relu'))
        model.add(keras.layers.Dropout(rate=hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hyperparameter tuning with Keras Tuner
def tune_dnn_model(X, y):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        overwrite=True,
        directory='kt_dir',
        project_name='diabetes_dnn'
    )
    tuner.search(X, y, epochs=20, validation_split=0.2, verbose=0)
    return tuner.get_best_models(num_models=1)[0]

# DNN with tuned hyperparameters
def evaluate_tuned_dnn(X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = tune_dnn_model(X_train, y_train)
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        y_pred_proba = model.predict(X_val).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)

        results.append({
            'Fold': fold + 1,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'F1': f1_score(y_val, y_pred),
            'AUC': roc_auc_score(y_val, y_pred_proba)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('C:/Users/DCL/Desktop/Research Paper/dnn_results.csv', index=False)
    return results_df

# XGBoost with actual feature importance from the last fold
def evaluate_xgboost_with_feature_importance(X, y, feature_names, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = []
    final_model = None  # To store the last model

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        results.append({
            'Fold': fold + 1,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'F1': f1_score(y_val, y_pred),
            'AUC': roc_auc_score(y_val, y_pred_proba)
        })

        final_model = model  # Store the model from the last fold

    results_df = pd.DataFrame(results)
    results_df.to_csv('C:/Users/DCL/Desktop/Research Paper/xgb_results.csv', index=False)

    # Get actual feature importance from the final fold's model
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue=None)
    plt.title('Feature Importance from Final XGBoost Model')
    plt.tight_layout()
    plt.savefig('C:/Users/DCL/Desktop/Research Paper/xgb_feature_importance.png')

    return results_df, importance_df


# Run evaluations
dnn_results = evaluate_tuned_dnn(X_scaled, y)
xgb_results, feature_importance = evaluate_xgboost_with_feature_importance(X_scaled, y, X.columns.tolist())

