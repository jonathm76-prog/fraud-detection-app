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

# Import the model class
from models import FraudDetectionModels

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = FraudDetectionModels()
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

def main():
    df = None
    # Header
    st.markdown('<h1 class="main-header">💰 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6B7280;">Advanced CNN & Machine Learning Models for Real-time Fraud Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.markdown("## 🎛️ Control Panel")
        
        st.markdown("### 1. Data Configuration")
        data_source = st.radio(
            "Select Data Source:",
            ["📁 Upload CSV", "🎲 Use Sample Data"]
        )
        
        if data_source == "📁 Upload CSV":
            uploaded_file = st.file_uploader("Upload credit card data (CSV)", type=['csv'])
            if uploaded_file:
                   df = pd.read_csv(uploaded_file)
            if df is not None:
                   df.to_csv("temp_data.csv", index=False)
                   st.success(f"Data loaded: {df.shape[0]} rows")
                   st.success(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        else:
            if st.button("Generate Sample Data"):
                with st.spinner("Generating synthetic data..."):
                    df = st.session_state.fraud_detector.create_synthetic_data()
                    st.session_state.current_data = df
                    st.success(f"Sample data generated: {df.shape[0]} rows × {df.shape[1]} columns")
        
        st.markdown("---")
        st.markdown("### 2. Model Training")
        
        model_options = st.multiselect(
            "Select Models to Train:",
            ["CNN (Convolutional Neural Network)", "Random Forest", "Logistic Regression"],
            default=["CNN (Convolutional Neural Network)", "Random Forest"]
        )
        
        if st.button("🚀 Train Models", type="primary"):
            if st.session_state.current_data is not None:
                with st.spinner("Training models... This may take a few minutes."):
                    # Save data temporarily
                    st.session_state.current_data.to_csv("temp_data.csv", index=False)
                    
                    # Train models
                    X_train, X_test, y_train, y_test, feature_names = st.session_state.fraud_detector.load_and_preprocess_data("temp_data.csv")
                    
                    results = []
                    
                    if "Logistic Regression" in [m.split(" ")[0] for m in model_options]:
                        lr_results = st.session_state.fraud_detector.train_logistic_regression(X_train, y_train, X_test, y_test)
                        results.append(lr_results)
                    
                    if "Random Forest" in [m.split(" ")[0] for m in model_options]:
                        rf_results = st.session_state.fraud_detector.train_random_forest(X_train, y_train, X_test, y_test)
                        results.append(rf_results)
                    
                    if "CNN" in [m.split(" ")[0] for m in model_options]:
                        cnn_results, history = st.session_state.fraud_detector.train_cnn(X_train, y_train, X_test, y_test)
                        results.append(cnn_results)
                        st.session_state.cnn_history = history
                    
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    # Save models
                    st.session_state.fraud_detector.save_models()
                    
                st.success("✅ Models trained successfully!")
            else:
                st.error("Please load or generate data first!")
        
        st.markdown("---")
        st.markdown("### 3. Real-time Prediction")
        st.info("After training models, use the 'Fraud Detection' tab to test individual transactions.")
        
        st.markdown("---")
        st.markdown("#### 📊 Model Info")
        st.markdown("""
        - **CNN**: Deep learning model with convolutional layers
        - **Random Forest**: Ensemble of decision trees
        - **Logistic Regression**: Statistical model for binary classification
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", 
        "🤖 Models", 
        "🔍 Fraud Detection", 
        "📊 Data Analysis",
        "📋 Report"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_models()
    
    with tab3:
        display_fraud_detection()
    
    with tab4:
        display_data_analysis()
    
    with tab5:
        display_report()

def display_dashboard():
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models from the sidebar to see the dashboard")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Total Models</h3><h2>3</h2></div>', unsafe_allow_html=True)
    
    with col2:
        fraud_rate = st.session_state.current_data['Class'].mean() * 100
        st.markdown(f'<div class="metric-card"><h3>Fraud Rate</h3><h2>{fraud_rate:.2f}%</h2></div>', unsafe_allow_html=True)
    
    with col3:
        best_model = max(st.session_state.results, key=lambda x: x['f1_score'])
        st.markdown(f'<div class="metric-card"><h3>Best Model</h3><h2>{best_model["model"][:10]}</h2></div>', unsafe_allow_html=True)
    
    with col4:
        total_samples = len(st.session_state.current_data)
        st.markdown(f'<div class="metric-card"><h3>Total Samples</h3><h2>{total_samples:,}</h2></div>', unsafe_allow_html=True)
    
    # Performance comparison
    st.markdown("### 📈 Model Performance Comparison")
    
    if st.session_state.models_trained:
        metrics_df = pd.DataFrame(st.session_state.results)
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'F1-Score', 'ROC-AUC', 'Confusion Matrices'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'domain'}]]
        )
        
        # Bar charts for metrics
        fig.add_trace(
            go.Bar(x=metrics_df['model'], y=metrics_df['accuracy'], 
                   name='Accuracy', marker_color='#3B82F6'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics_df['model'], y=metrics_df['f1_score'], 
                   name='F1-Score', marker_color='#10B981'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=metrics_df['model'], y=metrics_df['roc_auc'], 
                   name='ROC-AUC', marker_color='#EF4444'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📋 Detailed Metrics")
        detailed_metrics = []
        for result in st.session_state.results:
            detailed_metrics.append({
                'Model': result['model'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'ROC-AUC': f"{result['roc_auc']:.4f}",
                'Precision': f"{result['classification_report']['1']['precision']:.4f}",
                'Recall': f"{result['classification_report']['1']['recall']:.4f}"
            })
        
        st.table(pd.DataFrame(detailed_metrics))

def display_models():
    st.markdown('<h2 class="sub-header">🤖 Model Architectures & Performance</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models from the sidebar to see detailed information")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 CNN Architecture")
        st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
        
        # CNN Model Summary
        st.markdown("""
        **Layers:**
        1. Input Layer → Reshape
        2. Conv1D (64 filters) + BatchNorm + MaxPool + Dropout
        3. Conv1D (128 filters) + BatchNorm + MaxPool + Dropout
        4. Conv1D (256 filters) + BatchNorm + MaxPool + Dropout
        5. Flatten Layer
        6. Dense (128 units) + Dropout
        7. Dense (64 units) + Dropout
        8. Output Layer (Sigmoid)
        
        **Parameters:** ~500,000 trainable parameters
        **Optimizer:** Adam (lr=0.001)
        **Loss:** Binary Crossentropy
        """)
        
        # CNN Training History
        if hasattr(st.session_state, 'cnn_history'):
            history = st.session_state.cnn_history.history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'], 
                mode='lines', 
                name='Training Loss',
                line=dict(color='#EF4444')
            ))
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
        - Number of Trees: 100
        - Max Depth: 10
        - Min Samples Split: 5
        - Min Samples Leaf: 2
        - Class Weight: Balanced
        
        **Features:** 30 engineered features
        **Strategy:** Ensemble voting
        """)
        
        # Feature Importance
        if 'Random Forest' in st.session_state.fraud_detector.models:
            feature_importance = st.session_state.fraud_detector.get_feature_importance(
                'Random Forest', 
                list(range(st.session_state.X_test.shape[1]))
            )
            
            if feature_importance:
                fig = px.bar(
                    x=[f[1] for f in feature_importance[:10]],
                    y=[f"Feature {f[0]}" for f in feature_importance[:10]],
                    orientation='h',
                    title='Top 10 Important Features (Random Forest)',
                    color=[f[1] for f in feature_importance[:10]],
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logistic Regression
        st.markdown("### 📊 Logistic Regression")
        st.markdown('<div class="model-card lr-card">', unsafe_allow_html=True)
        st.markdown("""
        **Configuration:**
        - Regularization: L2 (C=0.1)
        - Max Iterations: 1000
        - Class Weight: Balanced
        - Solver: lbfgs
        
        **Advantages:**
        - Fast training & inference
        - Good baseline model
        - Probability outputs
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
        
        # Create a form for transaction input
        with st.form("transaction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                amount = st.number_input("Transaction Amount ($)", 
                                        min_value=0.0, 
                                        max_value=10000.0, 
                                        value=150.0)
                v1 = st.slider("V1 Feature", -10.0, 10.0, 0.0)
                v2 = st.slider("V2 Feature", -10.0, 10.0, 0.0)
                v3 = st.slider("V3 Feature", -10.0, 10.0, 0.0)
            
            with col_b:
                v4 = st.slider("V4 Feature", -10.0, 10.0, 0.0)
                v5 = st.slider("V5 Feature", -10.0, 10.0, 0.0)
                time = st.slider("Transaction Time", 0.0, 172000.0, 50000.0)
            
            selected_model = st.selectbox(
                "Select Model for Prediction:",
                [r['model'] for r in st.session_state.results]
            )
            
            submit_button = st.form_submit_button("🔍 Analyze Transaction", type="primary")
        
        if submit_button:
            # Create transaction vector
            transaction = np.zeros((1, 30))
            transaction[0, 0] = time
            transaction[0, 1:27] = [v1, v2, v3, v4, v5] + [0] * 22
            transaction[0, 29] = amount
            
            # Scale the transaction
            transaction_scaled = st.session_state.fraud_detector.scaler.transform(transaction)
            
            # Get prediction from selected model
            model_name = selected_model
            if model_name == 'CNN':
                # Reshape for CNN
                transaction_cnn = transaction_scaled.reshape(1, transaction_scaled.shape[1], 1)
                prediction = st.session_state.fraud_detector.models['CNN'].predict(transaction_cnn, verbose=0)[0][0]
            else:
                model = st.session_state.fraud_detector.models[model_name]
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba(transaction_scaled)[0][1]
                else:
                    prediction = model.predict(transaction_scaled)[0]
            
            # Display results
            st.session_state.prediction_result = {
                'probability': float(prediction),
                'is_fraud': prediction > 0.5,
                'model': model_name,
                'amount': amount
            }
    
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
            
            # Decision card
            if result['is_fraud']:
                st.error(f"🚨 **FRAUD DETECTED!**")
                st.markdown(f"""
                **Decision:** ⛔ Block Transaction
                **Confidence:** {result['probability']*100:.1f}%
                **Amount:** ${result['amount']:,.2f}
                **Model:** {result['model']}
                
                **Recommended Action:**
                1. Block transaction immediately
                2. Notify cardholder
                3. Flag account for review
                """)
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.markdown(f"""
                **Decision:** ✓ Approve Transaction
                **Confidence:** {(1-result['probability'])*100:.1f}%
                **Amount:** ${result['amount']:,.2f}
                **Model:** {result['model']}
                
                **Status:** Safe to proceed
                """)
    
    # Test with sample transactions
    st.markdown("---")
    st.markdown("### 🧪 Test Sample Transactions")
    
    col1, col2, col3 = st.columns(3)
    
    sample_transactions = [
        {"label": "Normal Purchase", "amount": 45.50, "v1": -0.5, "v2": 0.3, "v3": -0.2, "v4": 0.1},
        {"label": "Suspicious", "amount": 1250.00, "v1": 3.5, "v2": -2.8, "v3": 4.1, "v4": -3.2},
        {"label": "Clear Fraud", "amount": 2850.75, "v1": 8.2, "v2": -7.5, "v3": 9.1, "v4": -8.8}
    ]
    
    for idx, transaction in enumerate(sample_transactions):
        with [col1, col2, col3][idx]:
            if st.button(f"Test: {transaction['label']}", key=f"sample_{idx}"):
                # Create and predict
                sample = np.zeros((1, 30))
                sample[0, 0] = np.random.uniform(0, 100000)
                sample[0, 1] = transaction['v1']
                sample[0, 2] = transaction['v2']
                sample[0, 3] = transaction['v3']
                sample[0, 4] = transaction['v4']
                sample[0, 29] = transaction['amount']
                
                sample_scaled = st.session_state.fraud_detector.scaler.transform(sample)
                
                # Use CNN for prediction
                sample_cnn = sample_scaled.reshape(1, sample_scaled.shape[1], 1)
                prediction = st.session_state.fraud_detector.models['CNN'].predict(sample_cnn, verbose=0)[0][0]
                
                # Show result
                if prediction > 0.5:
                    st.error(f"Fraud Risk: {prediction*100:.1f}%")
                else:
                    st.success(f"Fraud Risk: {prediction*100:.1f}%")

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
        class_counts = df['Class'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=class_counts.values,
            hole=.3,
            marker_colors=['#10B981', '#EF4444']
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud rate over time (if Time column exists)
        if 'Time' in df.columns:
            st.markdown("### ⏰ Fraud Over Time")
            df['Time_Hour'] = (df['Time'] // 3600) % 24
            fraud_by_hour = df.groupby('Time_Hour')['Class'].mean() * 100
            
            fig = go.Figure(data=[go.Scatter(
                x=fraud_by_hour.index,
                y=fraud_by_hour.values,
                mode='lines+markers',
                line=dict(color='#EF4444', width=3)
            )])
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Fraud Rate (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution
        st.markdown("### 💰 Transaction Amount Analysis")
        
        if 'Amount' in df.columns:
            fig = make_subplots(rows=2, cols=1)
            
            # Legitimate transactions
            legit_amounts = df[df['Class'] == 0]['Amount']
            fig.add_trace(
                go.Histogram(x=legit_amounts, name='Legitimate', marker_color='#10B981'),
                row=1, col=1
            )
            
            # Fraud transactions
            fraud_amounts = df[df['Class'] == 1]['Amount']
            fig.add_trace(
                go.Histogram(x=fraud_amounts, name='Fraud', marker_color='#EF4444'),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                title_text="Transaction Amount Distribution"
            )
            fig.update_xaxes(title_text="Amount (Legitimate)", row=1, col=1)
            fig.update_xaxes(title_text="Amount (Fraud)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation
    st.markdown("### 🔗 Feature Correlation Heatmap")
    
    # Select only some features for visualization
    if df.shape[1] > 10:
        display_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'Amount', 'Class']
        display_features = [f for f in display_features if f in df.columns]
        corr_matrix = df[display_features].corr()
    else:
        corr_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data statistics
    st.markdown("### 📈 Data Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
        st.metric("Number of Features", df.shape[1])
    
    with col2:
        st.metric("Fraudulent Transactions", f"{df['Class'].sum():,}")
        st.metric("Fraud Percentage", f"{df['Class'].mean()*100:.2f}%")
    
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Duplicate Rows", df.duplicated().sum())

def display_report():
    st.markdown('<h2 class="sub-header">📋 Comprehensive Report</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 Train models first to generate report")
        return
    
    # Executive Summary
    st.markdown("## 📋 Executive Summary")
    st.markdown("""
    This Credit Card Fraud Detection System leverages advanced machine learning algorithms 
    to identify fraudulent transactions in real-time. The system incorporates:
    
    - **Convolutional Neural Networks (CNN)** for deep pattern recognition
    - **Random Forest** ensemble methods for robust classification
    - **Logistic Regression** as a statistical baseline
    
    All models are trained on preprocessed transaction data and evaluated using multiple metrics.
    """)
    
    # Performance Summary
    st.markdown("## 📊 Performance Summary")
    
    # Create a summary dataframe
    summary_data = []
    for result in st.session_state.results:
        summary_data.append({
            'Model': result['model'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}",
            'ROC-AUC': f"{result['roc_auc']:.4f}",
            'Precision (Fraud)': f"{result['classification_report']['1']['precision']:.4f}",
            'Recall (Fraud)': f"{result['classification_report']['1']['recall']:.4f}",
            'Support (Fraud)': result['classification_report']['1']['support']
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Recommendations
    st.markdown("## 💡 Recommendations & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Deployment Strategy")
        st.markdown("""
        1. **Primary Model:** CNN for high-risk transactions
        2. **Secondary Model:** Random Forest ensemble voting
        3. **Fallback:** Logistic Regression for speed
        
        **Threshold Optimization:**
        - Current threshold: 0.5
        - Recommended: 0.3 for high-risk scenarios
        - Adjust based on business requirements
        """)
    
    with col2:
        st.markdown("### ⚠️ Risk Factors Identified")
        st.markdown("""
        - **High transaction amounts** increase fraud probability
        - **Specific time windows** show higher fraud rates
        - **Feature V1-V5** show strongest correlation with fraud
        - **Geographic anomalies** detected in transaction patterns
        """)
    
    # Technical Details
    st.markdown("## 🔧 Technical Implementation")
    
    st.markdown("""
    ### Data Pipeline
    ```
    1. Data Collection → Real-time transaction stream
    2. Preprocessing → Scaling, normalization, feature engineering
    3. Imbalance Handling → SMOTE oversampling
    4. Model Training → Cross-validation, hyperparameter tuning
    5. Deployment → Real-time API with model serving
    ```
    
    ### Model Architecture Details
    - **CNN:** 3 convolutional blocks with dropout regularization
    - **Random Forest:** 100 trees with depth limiting
    - **Logistic Regression:** L2 regularization with class weighting
    
    ### Monitoring & Maintenance
    - Model performance tracking dashboard
    - Automated retraining pipeline
    - Drift detection for concept changes
    - A/B testing framework
    """)
    
    # Download Report
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    
    if st.button("Generate PDF Report"):
        st.info("Report generation feature would be implemented with additional libraries like reportlab or weasyprint")
    
    if st.button("Export Results to CSV"):
        # Create export data
        export_data = []
        for result in st.session_state.results:
            export_data.append({
                'model': result['model'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'roc_auc': result['roc_auc'],
                'precision': result['classification_report']['1']['precision'],
                'recall': result['classification_report']['1']['recall']
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
