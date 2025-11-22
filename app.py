import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Import RandomForestRegressor for type hint (though it's loaded from pickle)
from sklearn.ensemble import RandomForestRegressor 

# --- CRITICAL: MODEL & PREPROCESSING SETUP ---
@st.cache_data
def load_data_and_setup_model():
    """
    Loads data, fits the required preprocessors (Scaler, Encoder), and loads 
    the final trained model from the PKL file.
    """
    
    # 1. Load Raw Data (Needed for UI options and fitting Scaler/Encoder)
    try:
        df_raw = pd.read_csv("salaries.csv")
        df_raw.drop_duplicates(inplace=True)
    except FileNotFoundError:
        st.error("❌ Model file 'salaries.csv' not found. Please ensure it is committed.")
        st.stop()

    # 2. Setup Preprocessors
    categorical_cols = [
        "experience_level", "employment_type", "job_title", 
        "salary_currency", "employee_residence", "company_location", 
        "company_size"
    ]
    
    # NOTE: Your notebook used one LabelEncoder instance for all columns, 
    # which we replicate here for compatibility by fitting it on the last column only
    le = LabelEncoder()
    # Fit the LabelEncoder on one column to make it a valid object
    # For robust deployment, separate encoders/a pipeline should be used.
    le.fit(df_raw[categorical_cols[-1]]) 

    # 3. Create the Training Data Matrix (X) Structure
    # This prepares the data frame used to fit the StandardScaler
    X_df = df_raw.drop(columns=['salary', 'salary_currency', 'salary_in_usd'])
    
    # Temporarily encode the training data to fit the StandardScaler
    temp_df = X_df.copy()
    for col in categorical_cols:
        # NOTE: We must re-fit and transform the data here for the scaler fit.
        # This is inefficient but necessary if the fitted encoders weren't saved in the PKL.
        temp_df[col] = le.fit_transform(temp_df[col]) 

    X = temp_df.values
    
    # Split data to fit scaler ONLY on training data (Prevents Data Leakage)
    X_train, _, _, _ = train_test_split(X, df_raw['salary_in_usd'].values, test_size=0.20, random_state=42)
    
    # Fit StandardScaler
    ss = StandardScaler()
    ss.fit(X_train)

    # 4. Load the Trained Model (RandomForestRegressor)
    # Your final notebook cell saved the RF model alone under 'model_prediction.pkl'
    try:
        with open('model_prediction.pkl', 'rb') as file:
            rf_model = pickle.load(file)
            
            # Check if the loaded object is actually the model
            if not isinstance(rf_model, RandomForestRegressor):
                 st.error("❌ Error: PKL file does not contain a RandomForestRegressor object.")
                 st.stop()
                 
    except FileNotFoundError:
        st.error("❌ Model file 'model_prediction.pkl' not found. Please ensure it's committed to the repository.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model from PKL: {e}. Check requirements.txt pins.")
        st.stop()

    return rf_model, ss, le, X_df.columns.tolist(), df_raw

# Run the loader function
rf_model, scaler, le, FEATURE_NAMES, df_raw = load_data_and_setup_model()

# ==========================================================
# PAGE CONFIG AND STYLING (Code preserved from user request)
# ==========================================================
st.set_page_config(
    page_title="💼 Salary Prediction App",
    layout="wide",
    page_icon="💰"
)

st.markdown(
    """
    <style>
        /* Background gradient */
        .main {
            background: linear-gradient(145deg, #141E30 0%, #243B55 100%);
        }

        /* Text */
        h1, h2, h3, h4, h5, h6, div, p {
            color: #f1f1f1 !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Card design */
        .block-container {
            padding-top: 2rem;
        }

        .card {
            background: rgba(255, 255, 255, 0.07);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }

        /* Prediction text */
        .prediction-box {
            background: rgba(0, 255, 150, 0.15);
            border-left: 5px solid #00ff9d;
            padding: 15px;
            border-radius: 10px;
            font-size: 1.3rem;
            color: #00ff9d !important;
        }

        /* Button styling */
        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
        }

        .stButton>button:hover {
            background-color: #009acd;
            transform: scale(1.03);
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <h1 style='text-align:center; font-size: 3rem; margin-bottom: 0;'>
    💼 Salary Prediction App
    </h1>
    <p style='text-align:center; font-size: 1.2rem;'>
        Powered by D05 Batch ML Team (Model: Random Forest Regressor)
    </p>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# FEATURE INPUT UI
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔧 Enter Employee Details")

input_values = {}
cols = st.columns(2)

for i, col in enumerate(FEATURE_NAMES):
    with cols[i % 2]:
        # Determine if the feature is numerical or categorical based on the raw data
        if df_raw[col].dtype in [np.float64, np.int64]:
            min_val = float(df_raw[col].min())
            max_val = float(df_raw[col].max())
            median_val = float(df_raw[col].median())

            # Specific handling for 'work_year' and 'remote_ratio' which are effectively categorical
            if col in ['work_year', 'remote_ratio']:
                 input_values[col] = st.selectbox(
                    f"**{col.replace('_', ' ').title()}**", 
                    sorted(df_raw[col].unique().tolist()), 
                    index=sorted(df_raw[col].unique().tolist()).index(int(median_val))
                )
            else:
                input_values[col] = st.number_input(
                    f"**{col.replace('_', ' ').title()}**",
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val
                )

        else:
            # Handle categorical features
            choices = sorted(df_raw[col].unique().tolist())
            input_values[col] = st.selectbox(f"**{col.replace('_', ' ').title()}**", choices)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICT
# ==========================================================
if st.button("🚀 Predict Salary"):
    
    # 1. Convert user input to DataFrame
    input_df_raw = pd.DataFrame([input_values])
    
    # 2. Preprocessing (Label Encoding)
    # We must use the *fitted* encoder and assume the column order is preserved
    input_encoded = input_df_raw.copy()
    
    # Replicate the notebook's encoding logic using the fitted encoder
    for col in categorical_cols:
        if col in input_encoded.columns:
            # The LabelEncoder must be refitted to the column data to perform the transform
            # This is complex because the notebook used one LE for multiple columns.
            # For deployment, the LE needs to be robust, here we use a temporary fit 
            # to handle the single input row while preserving the structure.
            
            # Create a combined list of all unique values seen by the encoder 
            # (assuming this logic was safe in the notebook)
            
            # --- Robust single-row Label Encoding ---
            # Create a dictionary mapping the categorical columns to their original values
            # and transform them using the correct encoder
            
            # NOTE: Since the encoder was re-fitted in the notebook, we must re-fit it here 
            # on the full unique set of values to ensure it recognizes the input category.
            le_temp = LabelEncoder()
            le_temp.fit(df_raw[col]) # Fit on ALL historical values
            input_encoded[col] = le_temp.transform(input_encoded[col])
        # ---------------------------------------

    # 3. Scale the input data using the loaded StandardScaler
    # Ensure the column order is correct before scaling
    input_features = input_encoded[FEATURE_NAMES]
    input_scaled = scaler.transform(input_features)
    
    # 4. Make Prediction
    prediction = rf_model.predict(input_scaled)[0]

    st.balloons()
    
    st.markdown(
        f"""
        <div class='prediction-box'>
        <strong>Estimated Salary (USD):</strong> $<span style='color:#00ff9d;'>{prediction:,.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 0.9rem;'>Created with ❤️ in Streamlit</p>
    """,
    unsafe_allow_html=True
)