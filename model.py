# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
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
# Calculate Q1 (25th percentile) and Q3 (75th percentile) for each feature
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Define a condition for identifying outliers
outliers_condition = ((X < (Q1 - 1.3 * IQR)) | (X > (Q3 + 1.3 * IQR)))

# Remove rows where any feature is an outlier
X_no_outliers = X[~outliers_condition.any(axis=1)]
y_no_outliers = y[~outliers_condition.any(axis=1)]

# Check the shape after removing outliers
print("Original dataset shape:", X.shape)
print("Dataset shape after removing outliers:", X_no_outliers.shape)

# Step 2: Feature Engineering (on the filtered data)
X_no_outliers['LubOilPressure_LubOilTemp'] = X_no_outliers['Lub oil pressure'] * X_no_outliers['lub oil temp']
X_no_outliers['CoolantPressure_CoolantTemp'] = X_no_outliers['Coolant pressure'] * X_no_outliers['Coolant temp']
X_no_outliers['LubOilTemp_CoolantTemp'] = X_no_outliers['lub oil temp'] * X_no_outliers['Coolant temp']
X_no_outliers['LubOilPressure_CoolantPressure'] = X_no_outliers['Lub oil pressure'] * X_no_outliers['Coolant pressure']
X_no_outliers['EngineRPM_LubOilTemp'] = X_no_outliers['Engine rpm'] * X_no_outliers['lub oil temp']
X_no_outliers['EngineRPM_CoolantTemp'] = X_no_outliers['Engine rpm'] * X_no_outliers['Coolant temp']
X_no_outliers['EngineRPM_LubOilPressure'] = X_no_outliers['Engine rpm'] * X_no_outliers['Lub oil pressure']
X_no_outliers['EngineRPM_CoolantPressure'] = X_no_outliers['Engine rpm'] * X_no_outliers['Coolant pressure']

# Check correlation to understand the new features
print(X_no_outliers.corr())

# Step 3: Data Preprocessing
# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_no_outliers)

# ADASYN sampling
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y_no_outliers)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: Model Fine-Tuning
# Grid search for KNN hyperparameters
param_grid = {
    'n_neighbors': list(range(1, 20)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

knn = KNeighborsClassifier()
# Initialize StratifiedKFold for cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

grid_search = GridSearchCV(knn, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Step 6: Model Evaluation
optimized_knn = grid_search.best_estimator_
y_pred = optimized_knn.predict(X_test)

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Optional: Cross-validation for robustness
cv_scores = cross_val_score(optimized_knn, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Step 7: Test Instance Prediction
# Define a test instance (input data for prediction)
# Replace the values below with actual test input values
test_instance = np.array([[50, 85, 0.5, 120, 200, 80]])  # Example test input

# Add interaction features to the test instance
lub_oil_pressure = test_instance[0, 0]
lub_oil_temp = test_instance[0, 1]
coolant_pressure = test_instance[0, 2]
coolant_temp = test_instance[0, 3]
engine_rpm = test_instance[0, 4]

# Add derived features to the test instance
test_instance = np.append(test_instance, [
    lub_oil_pressure * lub_oil_temp,            # LubOilPressure_LubOilTemp
    coolant_pressure * coolant_temp,            # CoolantPressure_CoolantTemp
    lub_oil_temp * coolant_temp,                # LubOilTemp_CoolantTemp
    lub_oil_pressure * coolant_pressure,        # LubOilPressure_CoolantPressure
    engine_rpm * lub_oil_temp,                  # EngineRPM_LubOilTemp
    engine_rpm * coolant_temp,                  # EngineRPM_CoolantTemp
    engine_rpm * lub_oil_pressure,              # EngineRPM_LubOilPressure
    engine_rpm * coolant_pressure               # EngineRPM_CoolantPressure
])

# Reshape and scale the test instance
test_instance = test_instance.reshape(1, -1)
test_instance_scaled = scaler.transform(test_instance)

# Predict the engine condition
predicted_condition = optimized_knn.predict(test_instance_scaled)

# Display the prediction
print("Predicted Engine Condition:", predicted_condition[0])
