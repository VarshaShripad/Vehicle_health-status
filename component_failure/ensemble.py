import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE  # For oversampling minority classes
import joblib  # For saving the model and other objects

# Suppress warnings from PCA
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("predictive_maintenance.csv")  # Replace with your dataset file name

# Step 1: Feature Engineering
# Extract statistical features
data['Temperature Difference [K]'] = data['Process temperature [K]'] - data['Air temperature [K]']
data['Energy'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']

# Encode categorical target column if necessary
if 'Failure Type' in data.columns:
    label_encoder = LabelEncoder()
    data['Failure Type'] = label_encoder.fit_transform(data['Failure Type'])

# Step 2: Outlier Removal
# Use Isolation Forest to detect and remove outliers
feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                   'Torque [Nm]', 'Tool wear [min]', 'Temperature Difference [K]', 'Energy']
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(data[feature_columns])
data = data[outliers == 1]  # Retain only non-outliers

# Step 3: Separate features and target
X = data[feature_columns]
y = data['Failure Type']  # Target

# Step 4: Dimensionality Reduction (Optional, using PCA)
pca = PCA(n_components=5)  # Reduce to 5 principal components
X = pca.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scaling Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Use SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 8: Define Ensemble Model with class_weight='balanced'
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
gb_model = GradientBoostingClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

ensemble_model = VotingClassifier(
    estimators=[('Random Forest', rf_model), ('Gradient Boosting', gb_model), ('Logistic Regression', lr_model)],
    voting='soft'  # Use 'soft' for probability-based voting
)

# Step 9: Train the Ensemble Model
ensemble_model.fit(X_train_res, y_train_res)

# Step 10: Evaluate the Model
y_pred = ensemble_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

# Step 11: Predict Failure Type for a New Instance
def predict_failure(instance):
    instance = np.array(instance).reshape(1, -1)  # Reshape instance to a 2D array
    instance_pca = pca.transform(instance)  # Apply PCA (no need to worry about feature names)
    instance_scaled = scaler.transform(instance_pca)  # Scale the instance
    prediction = ensemble_model.predict(instance_scaled)  # Make the prediction
    return label_encoder.inverse_transform(prediction)[0]

# Example: Predict for a test instance
test_instance = [300, 350, 1500, 50, 200, 50, 75_000]  # Replace with actual test data
predicted_failure = predict_failure(test_instance)
print(f"Predicted Failure Type: {predicted_failure}")

# Step 12: Save the Model and Other Objects
# Save the trained model, scaler, PCA, and label encoder to disk
joblib.dump(ensemble_model, 'ensemble_model.pkl')  # Save trained ensemble model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
joblib.dump(pca, 'pca.pkl')  # Save the PCA transformer
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder
joblib.dump(smote, 'smote.pkl')  # Save the SMOTE sampler (if used)

print("Model, Scaler, PCA, and Label Encoder have been saved.")

# The saved files can be used for prediction later on.
