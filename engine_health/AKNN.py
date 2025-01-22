# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN

# Load the dataset
# Replace 'engine_data.csv' with your file path
data = pd.read_csv('engine_data.csv')

# Inspect dataset
print(data.head())
print(data.info())

# Define features (X) and target (y)
X = data.drop(columns=['Engine Condition'])  # Drop the target column
y = data['Engine Condition']

# Step 1: Outlier Removal Using IQR Method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers_condition = ((X < (Q1 - 1.3 * IQR)) | (X > (Q3 + 1.3 * IQR)))
X_no_outliers = X[~outliers_condition.any(axis=1)]
y_no_outliers = y[~outliers_condition.any(axis=1)]

# Check the shape after removing outliers
print("Original dataset shape:", X.shape)
print("Dataset shape after removing outliers:", X_no_outliers.shape)

# Step 2: Feature Engineering
X_no_outliers['LubOilPressure_LubOilTemp'] = X_no_outliers['Lub oil pressure'] * X_no_outliers['lub oil temp']
X_no_outliers['CoolantPressure_CoolantTemp'] = X_no_outliers['Coolant pressure'] * X_no_outliers['Coolant temp']
X_no_outliers['LubOilTemp_CoolantTemp'] = X_no_outliers['lub oil temp'] * X_no_outliers['Coolant temp']
X_no_outliers['LubOilPressure_CoolantPressure'] = X_no_outliers['Lub oil pressure'] * X_no_outliers['Coolant pressure']
X_no_outliers['EngineRPM_LubOilTemp'] = X_no_outliers['Engine rpm'] * X_no_outliers['lub oil temp']
X_no_outliers['EngineRPM_CoolantTemp'] = X_no_outliers['Engine rpm'] * X_no_outliers['Coolant temp']
X_no_outliers['EngineRPM_LubOilPressure'] = X_no_outliers['Engine rpm'] * X_no_outliers['Lub oil pressure']
X_no_outliers['EngineRPM_CoolantPressure'] = X_no_outliers['Engine rpm'] * X_no_outliers['Coolant pressure']

# Train a Random Forest to get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_no_outliers, y_no_outliers)
feature_importances = rf.feature_importances_

# Scale features by their importances
X_weighted = X_no_outliers * feature_importances

# Scale the weighted features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled_weighted = scaler.fit_transform(X_weighted)

# Resampling with ADASYN on the scaled weighted data
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled_weighted, y_no_outliers)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Model Fine-Tuning
param_grid = {
    'n_neighbors': list(range(1, 20)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

knn = KNeighborsClassifier()
stratified_kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

grid_search = GridSearchCV(knn, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Step 5: Model Evaluation
optimized_knn = grid_search.best_estimator_
y_pred = optimized_knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Optional: Cross-validation for robustness
cv_scores = cross_val_score(optimized_knn, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Step 6: Test Instance Prediction
test_instance = np.array([[50, 85, 0.5, 120, 200, 80]])
lub_oil_pressure = test_instance[0, 0]
lub_oil_temp = test_instance[0, 1]
coolant_pressure = test_instance[0, 2]
coolant_temp = test_instance[0, 3]
engine_rpm = test_instance[0, 4]

test_instance = np.append(test_instance, [
    lub_oil_pressure * lub_oil_temp,
    coolant_pressure * coolant_temp,
    lub_oil_temp * coolant_temp,
    lub_oil_pressure * coolant_pressure,
    engine_rpm * lub_oil_temp,
    engine_rpm * coolant_temp,
    engine_rpm * lub_oil_pressure,
    engine_rpm * coolant_pressure
])

test_instance = test_instance.reshape(1, -1)
test_instance_scaled = scaler.transform(test_instance)

predicted_condition = optimized_knn.predict(test_instance_scaled)
print("Predicted Engine Condition:", predicted_condition[0])
