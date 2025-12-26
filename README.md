💳 Credit Card Fraud Detection SystemAn end-to-end professional machine learning solution to detect fraudulent credit card transactions in real-time. This project utilizes advanced Convolutional Neural Networks (CNN) and Random Forest algorithms, integrated into an interactive web dashboard.


🚀 Overview:





Fraudulent transactions are a major challenge for the financial industry. This system provides a robust tool for banks and financial analysts to:Analyze imbalanced datasets using advanced sampling techniques (SMOTE).Train multiple models (CNN & Random Forest) simultaneously.Simulate live transactions to predict fraud probability instantly.



✨ Key Features:







Interactive Web Interface: Built with Streamlit for a seamless user experience.Dual-Model Architecture: Compare performance between Deep Learning (CNN) and Traditional ML (Random Forest).Imbalance Handling: Integrated SMOTE (Synthetic Minority Over-sampling Technique) to handle rare fraud cases (0.17% of total data).Live Prediction Sliders: Adjust transaction features (V1-V28, Amount, Time) to see real-time fraud alerts.Performance Metrics: Detailed visualizations including Confusion Matrix, Loss/Accuracy Curves, and Classification Reports.📊 DatasetThe project uses the Kaggle Credit Card Fraud Detection Dataset:Samples: 284,807 transactions.Features: 30 numerical features (Time, Amount, and V1-V28 PCA components).Class Imbalance: Only 492 transactions are fraudulent (0.17%).🛠️ Installation & SetupClone the Repository:Bashgit clone 
cd fraud-detection-app
Install Dependencies:Bashpip install -r requirements.txt
Run the Application:Bashstreamlit run app.py





🧠 Project StructurePlaintext



├── app.py              # Main Streamlit UI and dashboard logic
├── models.py           # ML & Deep Learning model architectures (CNN, Random Forest)
├── requirements.txt    # List of required libraries
├── README.md           # Project documentation
└── .streamlit/         # Theme and configuration (optional)




📈 Performance ResultsModelPrecisionRecallF1-ScoreRandom Forest94%82%88%CNN89%85%87%Note:




Results may vary based on training parameters and SMOTE configuration.🛡️ Future EnhancementsExplainable AI (SHAP): Adding feature importance explanations for every single prediction.Batch Prediction: Capability to upload an entire Excel file for bulk fraud screening.API Integration: Developing a REST API for integration with existing banking software.🤝 ContributingContributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvement.📄 LicenseDistributed under the MIT License. See LICENSE for more information.
