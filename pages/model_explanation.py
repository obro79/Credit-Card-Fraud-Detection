import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Model Building Process", layout="wide")

# Title
st.title("Credit Card Fraud Detection: Model Building Process")

# Introduction
st.write("""
# Model Development Journey

This page provides a detailed walkthrough of how we built and optimized our credit card fraud detection model. 
We'll cover each step of the process, from data analysis to model deployment, with visualizations and explanations.
""")

# Data Analysis Section
st.header("1. Data Analysis")

st.subheader("Class Distribution")
class_dist_img = Image.open('visualizations/class_balance.png')
st.image(class_dist_img, caption="Distribution of Fraudulent vs Legitimate Transactions")
st.write("""
The dataset shows a significant class imbalance, which is typical in fraud detection:
- Legitimate transactions make up the vast majority of cases
- Fraudulent transactions are rare events
- This imbalance required special handling during model training
""")

st.subheader("Transaction Amount Analysis")
amount_dist_img = Image.open('visualizations/amount_distributions.png')
st.image(amount_dist_img, caption="Distribution of Transaction Amounts")
st.write("""
Key insights from transaction amount analysis:
- Fraudulent transactions tend to have different amount patterns compared to legitimate ones
- Most legitimate transactions are of relatively small amounts
- Fraudulent transactions often show unusual amount patterns
""")

st.subheader("Time Analysis")
time_analysis_img = Image.open('visualizations/time_analysis.png')
st.image(time_analysis_img, caption="Transaction Time Analysis")
st.write("""
Analysis of transaction timing revealed:
- Certain time periods show higher fraud activity
- Legitimate transaction volume follows expected daily patterns
- Time of transaction is a valuable feature for fraud detection
""")

# Feature Engineering Section
st.header("2. Feature Engineering")

st.subheader("Feature Correlations")
correlation_img = Image.open('visualizations/correlation_heatmap.png')
st.image(correlation_img, caption="Feature Correlation Heatmap")
st.write("""
Feature correlation analysis helped us:
- Identify redundant features
- Understand relationships between variables
- Remove highly correlated features to prevent multicollinearity
""")

st.subheader("Feature Importance")
importance_img = Image.open('visualizations/feature_importance.png')
st.image(importance_img, caption="Feature Importance Plot")
st.write("""
XGBoost feature importance analysis revealed:
- Which features contribute most to fraud detection
- The relative importance of different transaction characteristics
- Key features that drive the model's decisions

### Feature Descriptions:
1. V1-V28: Principal Component Analysis (PCA) features
   - These are transformed features from the original transaction data
   - PCA is used to protect customer privacy while maintaining predictive power
   - Each component (V1-V28) represents a mathematical combination of original features

2. Transaction-specific features:
   - Amount: Raw transaction amount
   - Amount_Log: Natural logarithm of the transaction amount (helps handle large variations)
   - Hour: Hour of the day when transaction occurred (derived from Time)
   - Trans_Velocity: Number of transactions per hour
   
3. Interaction features:
   - V1*V2: Interaction between first two PCA components
   - V3*V4: Interaction between third and fourth PCA components
""")

st.subheader("Feature Interactions")
interactions_img = Image.open('visualizations/interaction_features.png')
st.image(interactions_img, caption="Feature Interaction Analysis")
st.write("""
We analyzed feature interactions to:
- Understand complex relationships between features
- Identify compound patterns that indicate fraud
- Create new interaction features where beneficial
""")

# Model Performance Section
st.header("3. Model Performance")

st.subheader("ROC Curve")
roc_img = Image.open('visualizations/roc_curve.png')
st.image(roc_img, caption="Receiver Operating Characteristic (ROC) Curve")
st.write("""
The ROC curve shows:
- Strong model discrimination ability
- High true positive rate while maintaining low false positive rate
- Area Under Curve (AUC) indicates excellent model performance
""")

st.subheader("Precision-Recall Curve")
pr_img = Image.open('visualizations/precision_recall_curve.png')
st.image(pr_img, caption="Precision-Recall Curve")
st.write("""
The Precision-Recall curve demonstrates:
- Model's ability to balance precision and recall
- Performance across different classification thresholds
- Robust performance despite class imbalance
""")

st.subheader("Confusion Matrix")
conf_matrix_img = Image.open('visualizations/confusion_matrix.png')
st.image(conf_matrix_img, caption="Confusion Matrix")
st.write("""
The confusion matrix reveals:
- High accuracy in detecting both legitimate and fraudulent transactions
- Low false positive rate, minimizing unnecessary alerts
- Good balance between sensitivity and specificity
""")

# Transaction Velocity Analysis
st.header("4. Advanced Analysis")

st.subheader("Transaction Velocity")
velocity_img = Image.open('visualizations/transaction_velocity.png')
st.image(velocity_img, caption="Transaction Velocity Analysis")
st.write("""
Transaction velocity analysis shows:
- Patterns in transaction frequency
- Unusual spikes that might indicate fraud
- Temporal patterns in fraudulent behavior
""")

# Model Selection and Optimization
st.header("5. Model Selection and Optimization")
st.write("""
### Why XGBoost?
We chose XGBoost as our final model because:
1. Superior performance on imbalanced datasets
2. Ability to handle complex feature interactions
3. Excellent scalability and prediction speed
4. Built-in handling of missing values

### Optimization Process
The model was optimized through:
- Extensive hyperparameter tuning using cross-validation
- Class weight balancing to handle imbalanced data
- Feature selection based on importance scores
- Regular retraining with new data

### Deployment Considerations
The model was deployed with:
- Real-time prediction capabilities
- Scalable architecture for high transaction volumes
- Monitoring system for model performance
- Regular updates and maintenance schedule
""")

# Conclusion
st.header("Conclusion")
st.write("""
Our fraud detection model achieves:
- High accuracy in fraud detection
- Low false positive rate
- Fast prediction times
- Robust performance across different transaction types

The model continues to be monitored and improved based on:
- New fraud patterns
- Changes in transaction behavior
- Performance metrics
- User feedback
""")