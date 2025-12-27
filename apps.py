import streamlit as st
from models import FraudDetectionModels

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the display functions from your current file
# (You'll need to split your current file into two)
