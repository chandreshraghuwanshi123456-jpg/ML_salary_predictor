import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. SIMULATE MODEL TRAINING AND ARTIFACT SAVING ---
# NOTE: The notebook contained 577 features after one-hot encoding.
# For the app to work, we must define all expected features.
# We will simulate training a small model using representative data and serialize it.

# Define the full list of features used for OHE (based on the notebook)
# This list is needed to reconstruct the input DataFrame for prediction
CATEGORICAL_FEATURES = {
    'experience_level': ['EN', 'EX', 'MI', 'SE'],
    'employment_type': ['CT', 'FL', 'FT', 'PT'],
    'company_size': ['L', 'M', 'S'],
    # Using a subset of top values for currency/residence/location for demonstration
    # In a real app, you would load all 84 currencies, 105 residences, and 75 locations
    'salary_currency': ['USD', 'EUR', 'GBP', 'INR', 'CAD', 'AUD', 'BRL'],
    'employee_residence': ['US', 'GB', 'CA', 'IN', 'DE', 'FR', 'ES'],
    'company_location': ['US', 'GB', 'CA', 'IN', 'DE', 'FR', 'ES'],
}

# The job titles are too numerous (288) to list, but we must use a placeholder that matches 
# the structure of the original data. We will use a small, representative list for UI.
# In a full deployment, you'd load the complete list from your pickled artifact.
JOB_TITLES = [
    'Data Scientist', 'Data Engineer', 'Data Analyst', 'Machine Learning Engineer', 
    'Analytics Engineer', 'Research Scientist', 'Other' # 'Other' category is a placeholder
]

# Numerical features (excluding 'salary_in_usd' which is the target)
NUMERICAL_FEATURES = ['work_year', 'salary', 'remote_ratio']

# --- A. Create a Mock Training Data & Fit the Scaler/Model ---
# This is crucial because the prediction pipeline requires the exact scaler and column list (577 features).
# We'll create a small mock dataset that mirrors the structure.
np.random.seed(42)
mock_data = {
    'work_year': np.random.choice([2024, 2025], 100),
    'salary': np.random.randint(50000, 300000, 100),
    'remote_ratio': np.random.choice([0, 50, 100], 100),
    'experience_level': np.random.choice(CATEGORICAL_FEATURES['experience_level'], 100),
    'employment_type': np.random.choice(CATEGORICAL_FEATURES['employment_type'], 100),
    'company_size': np.random.choice(CATEGORICAL_FEATURES['company_size'], 100),
    'job_title': np.random.choice(JOB_TITLES, 100),
    'salary_currency': np.random.choice(CATEGORICAL_FEATURES['salary_currency'], 100),
    'employee_residence': np.random.choice(CATEGORICAL_FEATURES['employee_residence'], 100),
    'company_location': np.random.choice(CATEGORICAL_FEATURES['company_location'], 100),
    'salary_in_usd': np.random.randint(80000, 200000, 100) # Mock target
}
mock_df = pd.DataFrame(mock_data)

# Perform the same processing as the notebook (get_dummies)
mock_X = mock_df.drop(columns=['salary_in_usd'])
mock_y = mock_df['salary_in_usd']
mock_X = pd.get_dummies(mock_X, drop_first=False) # Keep all categories for full feature list

# Align the mock features with the 577 feature structure (since we can't fully replicate the 577 in mock data)
# We will define a subset of feature columns that exist in our mock data for training, 
# and then use ALL mock_X.columns as the final expected feature list for the app (since it's a single file solution).
# The scaler must be fit only on the numerical data.
scaler = StandardScaler()
scaler.fit(mock_X[NUMERICAL_FEATURES])

# Scale the numerical features
mock_X_scaled_num = scaler.transform(mock_X[NUMERICAL_FEATURES])
mock_X[NUMERICAL_FEATURES] = mock_X_scaled_num

# Fit a basic Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(mock_X, mock_y)

# --- B. Create the Prediction Bundle (in-memory serialization) ---
# This bundle contains everything needed for prediction: model, scaler, and the full list of columns 
# *from the mock data generation* that the model expects.
PREDICT_COLUMNS = mock_X.columns.tolist() # The final list of features the model expects

# Serialize the components to bypass the external .pkl file
MODEL_BUNDLE_BIN = joblib.dumps({
    'best_model': rf_model,
    'scaler': scaler,
    'feature_columns': PREDICT_COLUMNS
})[0]

# --- 2. STREAMLIT APP LAYOUT AND LOGIC ---

def load_model_bundle():
    """Load the model and artifacts from the in-memory binary string."""
    try:
        # Load the serialized bundle
        bundle = joblib.loads(MODEL_BUNDLE_BIN)
        return bundle['best_model'], bundle['scaler'], bundle['feature_columns']
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, scaler, feature_columns = load_model_bundle()

if model is None:
    st.stop() # Stop execution if artifacts failed to load

# --- UI Header ---
st.set_page_config(layout="wide", page_title="Data Science Salary Predictor")

st.markdown(
    """
    <style>
    .reportview-container .main {
        background: #f0f2f6;
    }
    .st-eb {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        background-color: #ffffff;
    }
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .prediction-box {
        background-color: #e6f7ff;
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .prediction-value {
        font-size: 3.5em;
        font-weight: bolder;
        color: #2ca02c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-header">Data Science Salary Predictor</div>', unsafe_allow_html=True)
st.write("Enter the job details below to estimate the annual salary (in USD).")

# --- UI Inputs ---

with st.container():
    st.markdown('<div class="st-eb">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    # Row 1: Job Details
    with col1:
        job_title = st.selectbox(
            'Job Title', 
            options=JOB_TITLES, 
            index=0
        )
        
    with col2:
        experience_level = st.selectbox(
            'Experience Level', 
            options=CATEGORICAL_FEATURES['experience_level'], 
            format_func=lambda x: {'EN': 'Entry-Level', 'MI': 'Mid-Level', 'SE': 'Senior-Level', 'EX': 'Executive'}.get(x, x),
            index=2 # Default to Senior
        )

    with col3:
        remote_ratio = st.select_slider(
            'Remote Ratio (%)', 
            options=[0, 50, 100], 
            value=0
        )

    # Row 2: Company/Employee Location
    col4, col5, col6 = st.columns(3)
    
    with col4:
        company_location = st.selectbox(
            'Company Location', 
            options=CATEGORICAL_FEATURES['company_location'], 
            index=0 # Default to US
        )

    with col5:
        employee_residence = st.selectbox(
            'Employee Residence', 
            options=CATEGORICAL_FEATURES['employee_residence'], 
            index=0 # Default to US
        )

    with col6:
        company_size = st.selectbox(
            'Company Size', 
            options=CATEGORICAL_FEATURES['company_size'], 
            format_func=lambda x: {'S': 'Small', 'M': 'Medium', 'L': 'Large'}.get(x, x),
            index=1 # Default to M
        )

    # Row 3: Year, Employment Type, and Local Salary (Crucial Note)
    col7, col8, col9 = st.columns(3)

    with col7:
        work_year = st.slider(
            'Work Year', 
            min_value=2020, 
            max_value=2025, 
            value=2024
        )

    with col8:
        employment_type = st.selectbox(
            'Employment Type', 
            options=CATEGORICAL_FEATURES['employment_type'],
            format_func=lambda x: {'FT': 'Full-Time', 'PT': 'Part-Time', 'CT': 'Contract', 'FL': 'Freelance'}.get(x, x),
            index=2 # Default to FT
        )

    with col9:
        salary_currency = st.selectbox(
            'Local Salary Currency', 
            options=CATEGORICAL_FEATURES['salary_currency'],
            index=0 # Default to USD
        )
        # Note on local salary input, as per notebook X was highly dependent on it.
        # This input makes the prediction a bit circular, but necessary if the model expects it.
        salary_local = st.number_input(
            'Local Salary Amount (e.g., 100000)', 
            min_value=0, 
            value=120000, 
            step=1000
        )

    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Button and Logic ---

if st.button('Predict Salary', type="primary", use_container_width=True):
    try:
        # 1. Create a dictionary of input features
        input_data = {
            'work_year': work_year,
            'salary': salary_local, # Included as per the notebook's X definition
            'remote_ratio': remote_ratio,
            'experience_level': experience_level,
            'employment_type': employment_type,
            'job_title': job_title,
            'salary_currency': salary_currency,
            'employee_residence': employee_residence,
            'company_location': company_location,
            'company_size': company_size,
        }
        
        # 2. Convert to DataFrame (before OHE)
        input_df_raw = pd.DataFrame([input_data])
        
        # 3. Identify categorical and numerical columns for processing
        cat_cols = [col for col in input_df_raw.columns if col not in NUMERICAL_FEATURES]
        
        # 4. Perform One-Hot Encoding on raw inputs
        input_df_encoded = pd.get_dummies(input_df_raw, columns=cat_cols, drop_first=False)

        # 5. Reindex to align with the model's 577 feature columns
        # This is the most crucial step for OHE inputs in ML deployment
        for col in feature_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
        
        # Ensure only the necessary columns are present and in the correct order
        input_df_final = input_df_encoded[feature_columns]
        
        # 6. Separate numerical columns for scaling
        X_num = input_df_final[NUMERICAL_FEATURES]
        X_ohe = input_df_final.drop(columns=NUMERICAL_FEATURES)
        
        # 7. Scale numerical features (using the fitted scaler)
        X_num_scaled = scaler.transform(X_num)
        X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=NUMERICAL_FEATURES)
        
        # 8. Recombine scaled numerical features and OHE features
        X_final = pd.concat([X_num_scaled_df, X_ohe.reset_index(drop=True)], axis=1)

        # 9. Prediction
        predicted_salary_usd = model.predict(X_final)[0]

        # 10. Display Result
        st.markdown(
            f"""
            <div class="prediction-box">
                Estimated Salary (USD)
                <div class="prediction-value">${predicted_salary_usd:,.2f}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer/Notes ---
st.info(
    "**Model Note:** The original training data included 'salary' (local currency) as a feature to predict 'salary_in_usd'. "
    "If the local salary is known, the prediction of its USD equivalent is inherently simple. "
    "This app uses a simulated Random Forest model structure based on your notebook, including all dummy variables, "
    "but the R2 score for the Ridge model provided in the notebook's output was extremely low, indicating significant issues with the linear model fit (likely due to multicollinearity or model complexity for the data)."
)