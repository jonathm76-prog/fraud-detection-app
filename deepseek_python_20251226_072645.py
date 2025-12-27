import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .model-card {
        border-left: 5px solid;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    .cnn-card { border-color: #EF4444; }
    .rf-card { border-color: #10B981; }
    .lr-card { border-color: #3B82F6; }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'cnn_history' not in st.session_state:
    st.session_state.cnn_history = None

# Define FraudDetectionModels class directly in app.py (no import needed)
class FraudDetectionModels:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the credit card fraud dataset"""
        try:
            # Load the data
            df = pd.read_csv(filepath)
            
            # Check if dataset has the expected structure
            if 'Class' not in df.columns:
                if 'class' in df.columns:
                    df.rename(columns={'class': 'Class'}, inplace=True)
                elif 'target' in df.columns:
                    df.rename(columns={'target': 'Class'}, inplace=True)
                else:
                    # Assume last column is target
                    target_col = df.columns[-1]
                    df = df.rename(columns={target_col: 'Class'})
            
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class'].astype(int)
            
            # Scale all features
            X_scaled = self.scaler.fit_transform(X)
            feature_names = X.columns.tolist()
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Handle class imbalance with SMOTE
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            except:
                X_train_res, y_train_res = X_train, y_train
            
            return X_train_res, X_test, y_train_res, y_test, feature_names
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return self.create_and_preprocess_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic credit card data for demonstration"""
        np.random.seed(42)
        n_samples = 5000  # Reduced for faster processing
        n_features = 30
        
        # Create feature names
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Create fraud cases (1%)
        fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        
        # Fraud patterns
        for idx in fraud_indices:
            X[idx, :5] += np.random.uniform(2, 4, 5)
            X[idx, 5:10] -= np.random.uniform(1, 3, 5)
        
        # Create target variable
        y = np.zeros(n_samples)
        y[fraud_indices] = 1
        
        # Add noise
        X += np.random.randn(*X.shape) * 0.1
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        df['Class'] = y
        df['Amount'] = np.exp(np.random.randn(n_samples) * 0.5 + 4)
        df.loc[fraud_indices, 'Amount'] *= np.random.uniform(2, 5, len(fraud_indices))
        
        return df
    
    def create_and_preprocess_synthetic_data(self):
        """Create and preprocess synthetic data"""
        df = self.create_synthetic_data()
        
        # Process directly without saving to file
        X = df.drop('Class', axis=1)
        y = df['Class'].astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
        feature_names = X.columns.tolist()
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Logistic Regression model"""
        try:
            from sklearn.linear_model import LogisticRegression
            lr_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.1,
                solver='liblinear'
            )
            
            lr_model.fit(X_train, y_train)
            
            y_pred = lr_model.predict(X_test)
            y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
            
            self.models['Logistic Regression'] = lr_model
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'Logistic Regression')
        except Exception as e:
            st.error(f"Error training Logistic Regression: {e}")
            return self._create_dummy_metrics('Logistic Regression')
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
            
            rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            self.models['Random Forest'] = rf_model
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'Random Forest')
        except Exception as e:
            st.error(f"Error training Random Forest: {e}")
            return self._create_dummy_metrics('Random Forest')
    
    def build_cnn_model(self, input_shape):
        """Build a 1D CNN model for tabular data"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Reshape
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential()
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
            
            # Flatten and Dense layers
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            return model
        except Exception as e:
            st.error(f"Error building CNN model: {e}")
            return None
    
    def train_cnn(self, X_train, y_train, X_test, y_test):
        """Train and evaluate CNN model"""
        try:
            # Reshape data for CNN
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build model
            cnn_model = self.build_cnn_model((X_train.shape[1],))
            
            if cnn_model is None:
                raise ValueError("Failed to build CNN model")
            
            # Train model with fewer epochs for faster processing
            history = cnn_model.fit(
                X_train_cnn, y_train,
                validation_split=0.2,
                epochs=10,
                batch_size=32,
                class_weight={0: 1., 1: 10.},
                verbose=0
            )
            
            # Predictions
            y_pred_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Store results
            self.models['CNN'] = cnn_model
            
            return self._evaluate_model(y_test, y_pred, y_pred_proba, 'CNN'), history
            
        except Exception as e:
            st.error(f"Error training CNN: {e}")
            dummy_metrics = self._create_dummy_metrics('CNN')
            return dummy_metrics, None
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        try:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
            
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            metrics = {
                'model': model_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report
            }
            
            return metrics
        except Exception as e:
            st.error(f"Error evaluating model {model_name}: {e}")
            return self._create_dummy_metrics(model_name)
    
    def _create_dummy_metrics(self, model_name):
        """Create dummy metrics for failed models"""
        return {
            'model': model_name,
            'accuracy': 0.5,
            'f1_score': 0.5,
            'roc_auc': 0.5,
            'classification_report': {
                '0': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0},
                '1': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 0},
                'accuracy': 0.5
            }
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6B7280;">Advanced CNN & Machine Learning Models for Real-time Fraud Detection</p>', unsafe_allow_html=True)
    
    # Initialize detector in session state
    if 'fraud_detector' not in st.session_state:
        st.session_state.fraud_detector = FraudDetectionModels()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        st.markdown("### 1. Data Configuration")
        data_source = st.radio(
            "Select Data Source:",
            ["📁 Upload CSV", "🎲 Use Sample Data"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload credit card data (CSV)", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_data = df
                    st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    with st.expander("Preview Data"):
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        else:
            if st.button("Generate Sample Data", key="generate_sample"):
                with st.spinner("Generating synthetic data..."):
                    try:
                        detector = FraudDetectionModels()
                        df = detector.create_synthetic_data()
                        st.session_state.current_data = df
                        st.success(f"✅ Sample data generated: {df.shape[0]} rows × {df.shape[1]} columns")
                    except Exception as e:
                        st.error(f"❌ Error generating sample data: {e}")
        
        st.markdown("---")
        st.markdown("### 2. Model Training")
        
        model_options = st.multiselect(
            "Select Models to Train:",
            ["CNN", "Random Forest", "Logistic Regression"],
            default=["CNN", "Random Forest"]
        )
        
        if st.button("🚀 Train Models", type="primary"):
            if st.session_state.current_data is not None:
                st.session_state.training_in_progress = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Data preprocessing
                    status_text.text("Step 1/4: Preprocessing data...")
                    progress_bar.progress(25)
                    
                    # Create temporary file for CSV
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                        st.session_state.current_data.to_csv(tmp.name, index=False)
                        temp_path = tmp.name
                    
                    # Initialize detector
                    detector = FraudDetectionModels()
                    
                    # Preprocess data
                    X_train, X_test, y_train, y_test, feature_names = detector.load_and_preprocess_data(temp_path)
                    
                    results = []
                    
                    # Step 2: Train selected models
                    status_text.text("Step 2/4: Training models...")
                    
                    if "Logistic Regression" in model_options:
                        progress_bar.progress(40)
                        lr_results = detector.train_logistic_regression(X_train, y_train, X_test, y_test)
                        results.append(lr_results)
                    
                    if "Random Forest" in model_options:
                        progress_bar.progress(60)
                        rf_results = detector.train_random_forest(X_train, y_train, X_test, y_test)
                        results.append(rf_results)
                    
                    if "CNN" in model_options:
                        progress_bar.progress(80)
                        cnn_results, history = detector.train_cnn(X_train, y_train, X_test, y_test)
                        results.append(cnn_results)
                        st.session_state.cnn_history = history
                    
                    # Step 3: Finalize
                    status_text.text("Step 3/4: Finalizing...")
                    progress_bar.progress(90)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    st.session_state.detector = detector
                    st.session_state.feature_names = feature_names
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    status_text.text("✅ Training completed!")
                    progress_bar.progress(100)
                    st.success("✅ Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                finally:
                    st.session_state.training_in_progress = False
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.error("⚠️ Please load or generate data first!")
        
        st.markdown("---")
        st.markdown("### 3. Real-time Prediction")
        st.info("After training models, use the tabs below to test transactions.")
        
        # Reset button
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "🤖 Models", 
        "🔍 Fraud Detection", 
        "📊 Data Analysis"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_models()
    
    with tab3:
        display_fraud_detection()
    
    with tab4:
        display_data_analysis()

def display_dashboard():
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models from the sidebar to see the dashboard")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_count = len(st.session_state.results) if st.session_state.results else 0
        st.markdown(f'<div class="metric-card"><h3>Total Models</h3><h2>{model_count}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            target_col = 'Class' if 'Class' in df.columns else ('class' if 'class' in df.columns else None)
            if target_col and target_col in df.columns:
                fraud_rate = df[target_col].mean() * 100
                st.markdown(f'<div class="metric-card"><h3>Fraud Rate</h3><h2>{fraud_rate:.2f}%</h2></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.results and len(st.session_state.results) > 0:
            best_model = max(st.session_state.results, key=lambda x: x['f1_score'])
            st.markdown(f'<div class="metric-card"><h3>Best Model</h3><h2>{best_model["model"][:15]}</h2></div>', unsafe_allow_html=True)
    
    with col4:
        if st.session_state.current_data is not None:
            total_samples = len(st.session_state.current_data)
            st.markdown(f'<div class="metric-card"><h3>Total Samples</h3><h2>{total_samples:,}</h2></div>', unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("### 📈 Model Performance Comparison")
    
    if st.session_state.results and len(st.session_state.results) > 0:
        model_names = [r['model'] for r in st.session_state.results]
        accuracies = [r['accuracy'] for r in st.session_state.results]
        f1_scores = [r['f1_score'] for r in st.session_state.results]
        roc_aucs = [r['roc_auc'] for r in st.session_state.results]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Accuracy', 'F1-Score', 'ROC-AUC'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(go.Bar(x=model_names, y=accuracies, name='Accuracy', marker_color='#3B82F6'), row=1, col=1)
        fig.add_trace(go.Bar(x=model_names, y=f1_scores, name='F1-Score', marker_color='#10B981'), row=1, col=2)
        fig.add_trace(go.Bar(x=model_names, y=roc_aucs, name='ROC-AUC', marker_color='#EF4444'), row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📋 Detailed Metrics")
        detailed_metrics = []
        for result in st.session_state.results:
            report = result['classification_report']
            if '1' in report:
                detailed_metrics.append({
                    'Model': result['model'],
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'ROC-AUC': f"{result['roc_auc']:.4f}",
                    'Precision': f"{report['1']['precision']:.4f}",
                    'Recall': f"{report['1']['recall']:.4f}"
                })
        
        st.dataframe(pd.DataFrame(detailed_metrics))
    else:
        st.warning("No model results available.")

def display_models():
    st.markdown('<h2 class="sub-header">🤖 Model Architectures & Performance</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models from the sidebar to see detailed information")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 CNN Architecture")
        st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
        
        st.markdown("""
        **Layers:**
        1. Input Layer → Reshape
        2. Conv1D (32 filters) + BatchNorm + MaxPool + Dropout
        3. Conv1D (64 filters) + BatchNorm + MaxPool + Dropout
        4. Flatten Layer
        5. Dense (64 units) + Dropout
        6. Dense (32 units) + Dropout
        7. Output Layer (Sigmoid)
        
        **Optimizer:** Adam (lr=0.001)
        **Loss:** Binary Crossentropy
        """)
        
        # CNN Training History
        if st.session_state.cnn_history is not None:
            history = st.session_state.cnn_history.history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'], 
                mode='lines', 
                name='Training Loss',
                line=dict(color='#EF4444')
            ))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'], 
                    mode='lines', 
                    name='Validation Loss',
                    line=dict(color='#3B82F6')
                ))
            fig.update_layout(
                title='CNN Training History',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Random Forest
        st.markdown("### 🌳 Random Forest")
        st.markdown('<div class="model-card rf-card">', unsafe_allow_html=True)
        st.markdown("""
        **Configuration:**
        - Number of Trees: 50
        - Max Depth: 8
        - Min Samples Split: 5
        - Class Weight: Balanced
        
        **Strategy:** Ensemble voting
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logistic Regression
        st.markdown("### 📊 Logistic Regression")
        st.markdown('<div class="model-card lr-card">', unsafe_allow_html=True)
        st.markdown("""
        **Configuration:**
        - Regularization: L2 (C=0.1)
        - Max Iterations: 1000
        - Class Weight: Balanced
        - Solver: liblinear
        
        **Advantages:**
        - Fast training & inference
        - Good baseline model
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def display_fraud_detection():
    st.markdown('<h2 class="sub-header">🔍 Real-time Fraud Detection</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models first to enable real-time detection")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Test Transaction")
        
        with st.form("transaction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                amount = st.number_input("Transaction Amount ($)", 
                                        min_value=0.0, 
                                        max_value=10000.0, 
                                        value=150.0,
                                        step=10.0)
                v1 = st.slider("V1 Feature", -10.0, 10.0, 0.0, 0.1)
                v2 = st.slider("V2 Feature", -10.0, 10.0, 0.0, 0.1)
            
            with col_b:
                v3 = st.slider("V3 Feature", -10.0, 10.0, 0.0, 0.1)
                v4 = st.slider("V4 Feature", -10.0, 10.0, 0.0, 0.1)
                time = st.slider("Transaction Time", 0.0, 172000.0, 50000.0, 1000.0)
            
            selected_model = st.selectbox(
                "Select Model for Prediction:",
                [r['model'] for r in st.session_state.results] if st.session_state.results else []
            )
            
            submit_button = st.form_submit_button("🔍 Analyze Transaction", type="primary")
        
        if submit_button and 'detector' in st.session_state:
            try:
                # Create transaction vector
                n_features = len(st.session_state.feature_names) if hasattr(st.session_state, 'feature_names') else 30
                transaction = np.zeros((1, n_features))
                
                # Map features
                transaction[0, 0] = time if n_features > 0 else 0
                if n_features > 1: transaction[0, 1] = v1
                if n_features > 2: transaction[0, 2] = v2
                if n_features > 3: transaction[0, 3] = v3
                if n_features > 4: transaction[0, 4] = v4
                
                # Scale and predict
                transaction_scaled = st.session_state.detector.scaler.transform(transaction)
                
                if 'CNN' in selected_model and 'CNN' in st.session_state.detector.models:
                    transaction_cnn = transaction_scaled.reshape(1, transaction_scaled.shape[1], 1)
                    prediction = st.session_state.detector.models['CNN'].predict(transaction_cnn, verbose=0)[0][0]
                else:
                    model = st.session_state.detector.models.get(selected_model)
                    if model and hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba(transaction_scaled)[0][1]
                    else:
                        prediction = 0.0
                
                st.session_state.prediction_result = {
                    'probability': float(prediction),
                    'is_fraud': prediction > 0.5,
                    'model': selected_model,
                    'amount': amount
                }
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            # Fraud probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['probability'] * 100,
                title = {'text': "Fraud Probability (%)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            if result['is_fraud']:
                st.error(f"🚨 **FRAUD DETECTED!**")
                st.markdown(f"""
                **Decision:** ⛔ Block Transaction
                **Confidence:** {result['probability']*100:.1f}%
                **Amount:** ${result['amount']:,.2f}
                **Model:** {result['model']}
                """)
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.markdown(f"""
                **Decision:** ✓ Approve Transaction
                **Confidence:** {(1-result['probability'])*100:.1f}%
                **Amount:** ${result['amount']:,.2f}
                **Model:** {result['model']}
                """)

def display_data_analysis():
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Visualization</h2>', unsafe_allow_html=True)
    
    if st.session_state.current_data is None:
        st.info("👈 Load or generate data from the sidebar")
        return
    
    df = st.session_state.current_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Class distribution
        st.markdown("### 📊 Class Distribution")
        
        target_col = None
        for col in ['Class', 'class', 'target', 'is_fraud']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            class_counts = df[target_col].value_counts()
            labels = ['Legitimate', 'Fraud'] if len(class_counts) == 2 else [f'Class {i}' for i in class_counts.index]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=class_counts.values,
                hole=.3,
                marker_colors=['#10B981', '#EF4444'][:len(class_counts)]
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No target column found for class distribution")
    
    with col2:
        # Amount distribution
        st.markdown("### 💰 Transaction Amount Analysis")
        
        if 'Amount' in df.columns:
            fig = px.histogram(df, x='Amount', nbins=50, title="Transaction Amount Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Data statistics
    st.markdown("### 📈 Data Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
        st.metric("Number of Features", df.shape[1])
    
    with col2:
        if target_col:
            fraud_count = df[target_col].sum()
            fraud_pct = (fraud_count/len(df))*100
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
            st.metric("Fraud Percentage", f"{fraud_pct:.2f}%")

if __name__ == "__main__":
    main()
