import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, f1_score, precision_score, recall_score)
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

# For CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, 
                                   Dense, Dropout, BatchNormalization, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

class FraudDetectionModels:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the credit card fraud dataset"""
        print("Loading dataset...")
        
        try:
            # Load the data
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape}")
            
            # Check if dataset has the expected structure
            if 'Class' not in df.columns:
                # Try to identify target column
                if 'class' in df.columns:
                    df.rename(columns={'class': 'Class'}, inplace=True)
                elif 'target' in df.columns:
                    df.rename(columns={'target': 'Class'}, inplace=True)
                else:
                    # Assume last column is target
                    target_col = df.columns[-1]
                    df = df.rename(columns={target_col: 'Class'})
                    print(f"Renamed column '{target_col}' to 'Class'")
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Class distribution:\n{df['Class'].value_counts()}")
            print(f"Fraud percentage: {df['Class'].mean()*100:.2f}%")
            
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class'].astype(int)
            
            # Check for 'Time' and 'Amount' columns for special processing
            if 'Time' in X.columns:
                X['Time'] = self._process_time_column(X['Time'])
            
            if 'Amount' in X.columns:
                # Scale the 'Amount' column separately first
                amount_scaler = StandardScaler()
                X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
            
            # Scale all features
            X_scaled = self.scaler.fit_transform(X)
            feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Handle class imbalance with SMOTE
            try:
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                print(f"Training set after SMOTE: {X_train_res.shape}")
                print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
                X_train_res, y_train_res = X_train, y_train
            
            return X_train_res, X_test, y_train_res, y_test, feature_names
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create synthetic data as fallback
            print("Creating synthetic data as fallback...")
            return self._create_and_preprocess_synthetic_data()
    
    def _process_time_column(self, time_series):
        """Convert time to cyclical features"""
        # Simple normalization for now
        if time_series.max() > 1e6:  # Likely in seconds
            return (time_series - time_series.mean()) / time_series.std()
        return time_series
    
    def create_synthetic_data(self):
        """Create synthetic credit card data for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        n_features = 30
        
        # Create feature names (V1-V28, Time, Amount)
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Make first 5 features more important for fraud detection
        fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        
        # Fraud patterns
        for idx in fraud_indices:
            X[idx, :5] += np.random.uniform(2, 4, 5)  # Make these features unusually high
            X[idx, 5:10] -= np.random.uniform(1, 3, 5)  # Make these unusually low
            X[idx, 25] = np.random.uniform(5, 10)  # High amount for fraud
        
        # Create target variable (1% fraud)
        y = np.zeros(n_samples)
        y[fraud_indices] = 1
        
        # Add some noise to make it realistic
        X += np.random.randn(*X.shape) * 0.1
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        df['Class'] = y
        
        # Make Amount column more realistic
        df['Amount'] = np.exp(np.random.randn(n_samples) * 0.5 + 4)  # Log-normal distribution
        df.loc[fraud_indices, 'Amount'] *= np.random.uniform(2, 10, len(fraud_indices))
        
        print(f"Synthetic data created: {df.shape}")
        print(f"Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        
        return df
    
    def _create_and_preprocess_synthetic_data(self):
        """Create and preprocess synthetic data in one go"""
        df = self.create_synthetic_data()
        
        # Save to temp file and process
        df.to_csv("temp_synthetic.csv", index=False)
        return self.load_and_preprocess_data("temp_synthetic.csv")
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Logistic Regression model"""
        print("\nTraining Logistic Regression...")
        try:
            lr_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.1,
                solver='liblinear'
            )
            
            lr_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = lr_model.predict(X_test)
            y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
            
            # Store results
            self.models['Logistic Regression'] = lr_model
            self.results['Logistic Regression'] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': lr_model
            }
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'Logistic Regression')
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
            return self._create_dummy_metrics('Logistic Regression')
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Random Forest model"""
        print("\nTraining Random Forest...")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # Store results
            self.models['Random Forest'] = rf_model
            self.results['Random Forest'] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': rf_model
            }
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'Random Forest')
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return self._create_dummy_metrics('Random Forest')
    
    def build_cnn_model(self, input_shape):
        """Build a 1D CNN model for tabular data"""
        try:
            model = Sequential()
            
            # Reshape for Conv1D
            model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
            
            # First Conv Block
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.3))
            
            # Second Conv Block
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.3))
            
            # Third Conv Block
            model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.4))
            
            # Flatten and Dense layers
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
            
            return model
        except Exception as e:
            print(f"Error building CNN model: {e}")
            return None
    
    def train_cnn(self, X_train, y_train, X_test, y_test):
        """Train and evaluate CNN model"""
        print("\nTraining CNN Model...")
        
        try:
            # Reshape data for CNN (add channel dimension)
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build model
            cnn_model = self.build_cnn_model((X_train.shape[1],))
            
            if cnn_model is None:
                raise ValueError("Failed to build CNN model")
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
            ]
            
            # Train model
            history = cnn_model.fit(
                X_train_cnn, y_train,
                validation_split=0.2,
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                class_weight={0: 1., 1: 10.},  # Higher weight for fraud class
                verbose=0
            )
            
            # Predictions
            y_pred_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Store results
            self.models['CNN'] = cnn_model
            self.results['CNN'] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': cnn_model,
                'history': history
            }
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'CNN'), history
            
        except Exception as e:
            print(f"Error training CNN: {e}")
            dummy_metrics = self._create_dummy_metrics('CNN')
            return dummy_metrics, None
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
            
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            metrics = {
                'model': model_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'classification_report': report
            }
            
            print(f"\n{model_name} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))
            
            return metrics
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            return self._create_dummy_metrics(model_name)
    
    def _create_dummy_metrics(self, model_name):
        """Create dummy metrics for failed models"""
        return {
            'model': model_name,
            'accuracy': 0.5,
            'f1_score': 0.5,
            'roc_auc': 0.5,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'classification_report': {
                '0': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0},
                '1': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0},
                'accuracy': 0.5,
                'macro avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0},
                'weighted avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0}
            }
        }
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        try:
            if model_name == 'Random Forest' and model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Return top features
                    top_features = [(feature_names[i] if i < len(feature_names) else f"Feature_{i}", 
                                   importances[i]) for i in indices[:20]]
                    return top_features
        except Exception as e:
            print(f"Error getting feature importance: {e}")
        return None
    
    def save_models(self, path='saved_models'):
        """Save trained models"""
        try:
            os.makedirs(path, exist_ok=True)
            
            for name, model in self.models.items():
                if name == 'CNN':
                    model.save(f'{path}/cnn_model.h5')
                    print(f"Saved CNN model to {path}/cnn_model.h5")
                else:
                    joblib.dump(model, f'{path}/{name.lower().replace(" ", "_")}.pkl')
                    print(f"Saved {name} model to {path}/{name.lower().replace(' ', '_')}.pkl")
            
            joblib.dump(self.scaler, f'{path}/scaler.pkl')
            print(f"\nAll models saved to {path}/")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, path='saved_models'):
        """Load trained models"""
        try:
            if os.path.exists(f'{path}/scaler.pkl'):
                self.scaler = joblib.load(f'{path}/scaler.pkl')
            
            # Load CNN model
            if os.path.exists(f'{path}/cnn_model.h5'):
                self.models['CNN'] = tf.keras.models.load_model(f'{path}/cnn_model.h5')
            
            # Load other models
            for model_file in os.listdir(path):
                if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
                    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                    self.models[model_name] = joblib.load(f'{path}/{model_file}')
            
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
