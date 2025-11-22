import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define categorical columns used for encoding (Corrected list from notebook logic)
CATEGORICAL_COLS = [
    "experience_level", "employment_type", "job_title", 
    "employee_residence", "company_location", "company_size"
]

# --- CRITICAL: MODEL LOADING AND INITIAL DATA SETUP ---
@st.cache_resource
def load_model_bundle():
    """Loads model bundle and raw data for UI options."""
    try:
        # Load the bundled dictionary (model, scaler, feature_names)
        with open('model_bundle.pkl', 'rb') as file:
            MODEL_BUNDLE = pickle.load(file)
            
        # Load raw data for UI options and encoding reference
        df_raw = pd.read_csv("salaries.csv")
        df_raw.drop_duplicates(inplace=True)
        
        # Prepare UI options from raw data
        UI_OPTIONS = {}
        for col in MODEL_BUNDLE['feature_names']:
            if col in df_raw.columns:
                # Store unique values as strings for consistent use in selectbox
                if df_raw[col].dtype == 'object' or col in ['work_year', 'remote_ratio']:
                     UI_OPTIONS[col] = sorted(df_raw[col].unique().astype(str).tolist())
                else:
                    UI_OPTIONS[col] = sorted(df_raw[col].unique().tolist())

        return MODEL_BUNDLE, df_raw, UI_OPTIONS

    except FileNotFoundError:
        st.error("❌ Required file 'model_bundle.pkl' or 'salaries.csv' not found in the repository. Ensure both are committed.")
        st.stop()
    except Exception as e:
        # Display the specific error that occurred during loading
        st.markdown(f"""
            <div style='color:red; background-color: #331111; padding: 10px; border-radius: 5px;'>
                ❌ Error during Model Loading (Check requirements.txt and file integrity): {e}
            </div>
            """, unsafe_allow_html=True)
        st.stop()

# Run the loader function once
MODEL_BUNDLE, df_raw, UI_OPTIONS = load_model_bundle()

# Extract components for direct use
rf_model = MODEL_BUNDLE['model']
scaler = MODEL_BUNDLE['scaler']
FEATURE_NAMES = MODEL_BUNDLE['feature_names']


# ==========================================================
# PAGE CONFIG AND STYLING 
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
            color: #f1f1f1 !important;
        }
        .prediction-box strong, .prediction-box span {
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

st.sidebar.title("⚙️ Controls")
st.sidebar.info("Adjust inputs and press *Predict Salary*")

# ==========================================================
# FEATURE INPUT UI
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔧 Enter Employee Details")

input_values = {}
cols = st.columns(2)

for i, col in enumerate(FEATURE_NAMES):
    with cols[i % 2]:
        display_name = col.replace('_', ' ').title()
        
        # Check if feature is categorical object type in the raw data
        is_categorical_object = col in df_raw.columns and df_raw[col].dtype == 'object'

        if is_categorical_object:
            choices = UI_OPTIONS.get(col, [])
            input_values[col] = st.selectbox(f"**{display_name}**", choices)
            
        else:
            if col in df_raw.columns:
                min_val = float(df_raw[col].min())
                max_val = float(df_raw[col].max())
                median_val = float(df_raw[col].median())
                
                # Use selectbox for discrete numeric features (work_year, remote_ratio)
                if col in ['work_year', 'remote_ratio']:
                     median_str = str(int(median_val))
                     choices = UI_OPTIONS[col]
                     
                     try:
                         default_index = choices.index(median_str)
                     except ValueError:
                         default_index = 0
                         
                     input_values[col] = st.selectbox(
                        f"**{display_name}**", 
                        choices, 
                        index=default_index
                    )
                else:
                    # Use number_input for other numeric features
                    input_values[col] = st.number_input(
                        f"**{display_name}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val
                    )
            else:
                 input_values[col] = st.number_input(f"**{display_name}**", value=0.0)


st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICT FUNCTION
# ==========================================================
if st.button("🚀 Predict Salary"):
    
    # 1. Convert user input dictionary to a DataFrame (single row)
    input_df_raw = pd.DataFrame([input_values])
    
    # 2. Preprocessing (Label Encoding)
    input_encoded = input_df_raw.copy()
    
    for col in CATEGORICAL_COLS:
        if col in input_encoded.columns:
            le_temp = LabelEncoder()
            # Fit on the entire training distribution (df_raw)
            le_temp.fit(df_raw[col].astype(str)) 
            # Transform the user's single input
            input_encoded[col] = le_temp.transform(input_encoded[col].astype(str))

    # 3. Align Feature Order (CRITICAL)
    # Reindex the input DataFrame to match the exact order of features the model was trained on
    try:
        input_features = input_encoded.reindex(columns=FEATURE_NAMES, fill_value=0)
    except KeyError as e:
        st.error(f"Feature mismatch error: {e}. Cannot align input columns with trained model features.")
        st.stop()
        
    # 4. Scale the input data using the loaded, fitted scaler
    input_scaled = scaler.transform(input_features)
    
    # 5. Make Prediction
    prediction = rf_model.predict(input_scaled)[0]

    st.balloons()
    
    # 6. Display Result
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