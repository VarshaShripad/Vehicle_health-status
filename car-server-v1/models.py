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
import joblib

class PredictiveMaintenanceModel:
    def __init__(self, model_file='ensemble_model.pkl', scaler_file='ensemble_scaler.pkl', pca_file='pca.pkl', label_encoder_file='label_encoder.pkl', smote_file='smote.pkl'):
        # Load saved models and preprocessing steps
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.pca = joblib.load(pca_file)
        self.label_encoder = joblib.load(label_encoder_file)
        self.smote = joblib.load(smote_file)
        
        print("Model and preprocessing steps loaded successfully.")


    def predict_failure(self, instance):
        """
        Predict failure type for a new instance.
        """
        instance = np.array(instance).reshape(1, -1)  # Reshape instance to a 2D array
        instance_pca = self.pca.transform(instance)  # Apply PCA
        instance_scaled = self.scaler.transform(instance_pca)  # Scale the instance
        prediction = self.model.predict(instance_scaled)  # Make the prediction
        return self.label_encoder.inverse_transform(prediction)[0]


class EngineConditionPredictor:
    def __init__(self, model_file='knn_model.pkl', scaler_file='aknn_scaler.pkl', feature_importances_file='feature_importances.pkl'):
        """
        Initializes the model, scaler, and feature importances from disk.
        """
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_importances = joblib.load(feature_importances_file)
        self.featureLabels=['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure',
       'lub oil temp', 'Coolant temp', 'LubOilPressure_LubOilTemp',
       'CoolantPressure_CoolantTemp', 'LubOilTemp_CoolantTemp',
       'LubOilPressure_CoolantPressure', 'EngineRPM_LubOilTemp',
       'EngineRPM_CoolantTemp', 'EngineRPM_LubOilPressure',
       'EngineRPM_CoolantPressure']
        print("Model, Scaler, and Feature Importances loaded successfully.")


    def predict_condition(self, instance):
        """
        Predicts the engine condition for a new instance.
        """
        # Extract values
        lub_oil_pressure = instance[0, 1]
        lub_oil_temp = instance[0, 4]
        coolant_pressure = instance[0, 3]
        coolant_temp = instance[0, 5]
        engine_rpm = instance[0, 0]

        # Add feature engineered columns to the test instance
        instance = np.append(instance, [
            lub_oil_pressure * lub_oil_temp,
            coolant_pressure * coolant_temp,
            lub_oil_temp * coolant_temp,
            lub_oil_pressure * coolant_pressure,
            engine_rpm * lub_oil_temp,
            engine_rpm * coolant_temp,
            engine_rpm * lub_oil_pressure,
            engine_rpm * coolant_pressure
        ])

        # Convert the test instance to a DataFrame with the same columns as the training data
        instance_df = pd.DataFrame(instance.reshape(1, -1), columns=self.featureLabels)

        # Scale the test instance using the same scaler fitted on the training data
        instance_scaled = self.scaler.transform(instance_df)

        # Make prediction
        predicted_condition = self.model.predict(instance_scaled)
        return predicted_condition[0]
'''   
model = PredictiveMaintenanceModel()  # This loads the model, scaler, pca, label_encoder, and smote into memory
test_instance = [300, 350, 1500, 50, 200, 50, 75_000]  # Replace with actual test data
predicted_failure = model.predict_failure(test_instance)
print(f"Predicted Failure Type: {predicted_failure}")



predictor = EngineConditionPredictor()  # This loads the model, scaler, and feature importances into memory
test_instance = np.array([[700, 2.493591821, 11.79092738, 3.178980794, 84.14416293, 81.6321865]])  # Example test data
predicted_condition = predictor.predict_condition(test_instance)
print(f"Predicted Engine Condition: {predicted_condition}")

'''