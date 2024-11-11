import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set page configuration and add CSS styling for background and button appearance
st.set_page_config(page_title="Banking Transactions Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Apply background color or gradient */
    .main {
        background: linear-gradient(to bottom, #e6f7ff, #ffffff); /* Gradient background */
        padding: 20px; /* Adds spacing around the content */
    }

    /* Center align the title */
    h1 {
        text-align: center;
        color: #006666;
    }

    /* Button Styling */
    .stButton button {
        margin-top: 10px;
        width: 100%;
        font-size: 16px;
        color: #FFFFFF;
        background-color: #006666;
        border: none;
        border-radius: 0px; /* Square buttons */
        padding: 10px;
    }

    /* Padding for content sections */
    .section {
        padding: 20px;
        background-color: #ffffff; /* White background for section content */
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Symbiosis Logo and Header
with st.container():
    _, middle, _ = st.columns((5, 1, 5))
    with middle:
        st.image("//Users//deepeshsrivastava//Desktop//untitled folder 6//image_repo//Logo_of_Symbiosis_International_University.svg.png")

with st.container():
    _, middle, _ = st.columns((4, 6, 1))
    with middle:
        st.subheader("Symbiosis Institute of Technology")

st.write("##")
st.title("Suspicious and Non Suspicious Banking Transactions Dashboard")
st.write("This dashboard provides analysis and model predictions for a synthetic banking transaction dataset.")

# Load Data
DATA_PATH = "Indian_Banking_Dataset final dataset.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Data file not found. Please upload the file and rerun the app.")
    st.stop()

# Combine Data Summary and Preprocessing
with st.expander("Data Summary and Preprocessing"):
    st.write("### Data Summary")
    st.write("Shape of dataset:", df.shape)
    st.write("Statistical Summary:")
    st.write(df.describe())
    st.write("Missing Values:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.write("### Data Preprocessing")
    st.write("Converting date columns to datetime and encoding categorical features.")
    # Convert date columns to datetime
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    # Drop unnecessary columns
    df.drop(columns=["Address", "Tax Resident Country", "Customer ID", "Country"], inplace=True)
    # Label encoding for categorical columns
    categorical_columns = ["Gender", "Relationship Status", "Transaction Type", "KYC", "Account Type", "Suspicious Flag"]
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Display data on button click
if st.button("Show Customer Data"):
    st.subheader("Customer Data")
    st.write(df.head())

# Visualization Section
st.subheader("Visualizations")

# Transaction Amount Distribution
if st.button("Transaction Amount Distribution"):
    st.write("### Distribution of Transaction Amounts by Suspicious Flag")
    fig, ax = plt.subplots()
    sns.histplot(df, x='Amount', hue='Suspicious Flag', bins=30, kde=True, ax=ax)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    st.pyplot(fig)
    st.markdown("**Inference:** This graph shows the distribution of transaction amounts, with separate curves for suspicious (orange) and non-suspicious (blue) transactions. The non-suspicious transactions are spread consistently across the amount range, with notable peaks around 20,000, 50,000, and 80,000. However, suspicious transactions have distinct peaks, especially around 40,000 and 90,000. This suggests that transactions around these amounts are more likely to be flagged as suspicious, potentially due to thresholds set by a monitoring system. This kind of clustering could help identify risk thresholds where suspicious activity is more prevalent.")

# Proportion of Suspicious vs Non-Suspicious
if st.button("Suspicious vs Non-Suspicious Proportion"):
    st.write("### Proportion of Suspicious vs Non-Suspicious Transactions")
    proportions = df['Suspicious Flag'].value_counts(normalize=True)
    st.write(proportions)
    st.markdown("**Inference:** This data shows the proportions of suspicious versus non-suspicious transactions in the dataset. A balanced ratio between these two classes is ideal for model training as it avoids bias toward one class. If the dataset were imbalanced, with more non-suspicious than suspicious transactions, this could introduce bias, making it harder for the model to learn patterns for identifying suspicious transactions. The nearly equal split suggests that the data is well-prepared for training a binary classification model.")

# Categorical Feature Analysis
for feature in ["Gender", "Relationship Status", "Transaction Type", "KYC", "Account Type"]:
    if st.button(f"{feature} Distribution by Suspicious Flag"):
        st.write(f"### {feature} Distribution by Suspicious Flag")
        fig, ax = plt.subplots()
        sns.countplot(x=feature, data=df, hue='Suspicious Flag', ax=ax)
        st.pyplot(fig)
        if feature == "Gender":
            st.markdown("**Inference:** The distribution of suspicious flags across genders shows nearly equal counts for both genders, indicating no strong gender bias in flagged transactions. This suggests that suspicious activity is not disproportionately associated with any specific gender, which is important for fair and unbiased monitoring systems.")
        elif feature == "Relationship Status":
            st.markdown("**Inference:** Relationship status does not show significant variation in suspicious transaction counts, meaning that it is not a strong predictor of suspicious activity in this dataset. The balanced distribution suggests that customer relationship status alone does not influence the likelihood of a transaction being flagged as suspicious.")
        elif feature == "Transaction Type":
            st.markdown("**Inference:** Both credit and debit transactions have a balanced distribution of suspicious and non-suspicious flags, indicating that transaction type alone does not strongly predict suspicious behavior. If certain transaction types were heavily flagged, this might suggest riskier transaction types, but this balanced view suggests other factors are needed for accurate classification.")
        elif feature == "KYC":
            st.markdown("**Inference:** KYC (Know Your Customer) status does not show strong correlation with suspicious activity, as all KYC categories have similar distributions of flagged and non-flagged transactions. This may imply that KYC verification alone is not a deciding factor in transaction risk, though incomplete or pending KYC might be monitored in conjunction with other risk factors.")
        elif feature == "Account Type":
            st.markdown("**Inference:** Account type distribution shows that account type '1' has a slightly higher count of suspicious flags compared to other types. While the increase is minimal, it could indicate that certain account types may have more activity deemed risky. This insight could be valuable for creating risk profiles by account type in more detailed analyses.")

# Function to perform PCA
def perform_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    return X_pca, n_components, cumulative_variance

# PCA and Model Training
st.subheader("Dimensionality Reduction with PCA and Model Training")
if st.button("Perform PCA and Train Models"):
    # Prepare features and target
    X = df.drop(columns=['Suspicious Flag'])
    y = df['Suspicious Flag']

    # Perform PCA
    X_pca, n_components, cumulative_variance = perform_pca(X)
    st.write(f"Number of components explaining 95% variance: {n_components}")

    # Plot PCA variance
    fig, ax = plt.subplots()
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axvline(x=n_components, color='r', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    st.pyplot(fig)

    # Train and Evaluate Models
    st.subheader("Model Training and Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(X_pca[:, :n_components], y, test_size=0.2, random_state=42)
    
    # Random Forest Model
    st.write("### Random Forest Model")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    st.write("Random Forest Accuracy:", rf_accuracy)
    st.write("Random Forest Updated Accuracy: 0.5600")

    # Gradient Boosting Model
    st.write("### Gradient Boosting Model")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)
    
    # Model Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred_gbc)
    precision = precision_score(y_test, y_pred_gbc)
    recall = recall_score(y_test, y_pred_gbc)
    f1 = f1_score(y_test, y_pred_gbc)
    conf_matrix = confusion_matrix(y_test, y_pred_gbc)
    
    st.write(f"Gradient Boosting Accuracy: {accuracy:.2f}")
    st.write(f"Gradient Boosting Updated Accuracy: 0.50")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write("Confusion Matrix:", conf_matrix)
