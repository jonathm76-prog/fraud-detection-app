import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

class FraudDetectionModels:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.smote = SMOTE(random_state=42)
        self.is_trained = False

    def prepare_data(self, df):
        # Splitting features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scaling features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Applying SMOTE to the training set to balance the classes
        X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model

    def predict(self, X_input):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        return self.model.predict(X_input)

    def predict_proba(self, X_input):
        return self.model.predict_proba(X_input)
