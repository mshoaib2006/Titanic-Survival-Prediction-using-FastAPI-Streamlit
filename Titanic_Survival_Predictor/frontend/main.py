import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE = "http://localhost:8001"  
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"
TEST_URL = f"{API_BASE}/test_prediction"

# Constants and mappings

MAPPINGS = {
    'class': {"First": 1, "Second": 2, "Third": 3},
    'gender': {"Male": "male", "Female": "female"},
    'embarked': {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"},
    'title': {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Other": "Other"}
}

def get_input_data():
    """Collect and transform form data into API-ready format"""
    return {
        'Pclass': MAPPINGS['class'][st.session_state.pclass],
        'Sex': MAPPINGS['gender'][st.session_state.gender],
        'Age': float(st.session_state.age),
        'SibSp': int(st.session_state.sibsp),
        'Parch': int(st.session_state.parch),
        'Fare': float(st.session_state.fare),
        'Embarked': MAPPINGS['embarked'][st.session_state.embarked],
        'Title': st.session_state.title,
        'HadCabin': bool(st.session_state.had_cabin),
        'FamilySize': st.session_state.sibsp + st.session_state.parch + 1,
        'IsAlone': bool(st.session_state.sibsp + st.session_state.parch == 0)
    }

def check_api_health():
    """Verify API connection"""
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None

# Sidebar with API status
with st.sidebar:
    st.header("API Status")
    api_healthy, health_data = check_api_health()
    
    if api_healthy:
        st.success("API Connected")
        if health_data:
            
            st.caption(f"Features: {', '.join(health_data.get('expected_features', []))}")
        
        if st.button("Test API", help="Send test prediction"):
            try:
                test_response = requests.post(TEST_URL, timeout=5)
                if test_response.status_code == 200:
                    test_data = test_response.json()
                    st.success(f"Test passed! Prediction: {test_data.get('interpretation', 'Unknown')}")
                else:
                    st.error(f"Test failed (Status {test_response.status_code})")
            except Exception as e:
                st.error(f"Test error: {str(e)}")
    else:
        st.error("API Unavailable")
        st.markdown("""
        Ensure FastAPI server is running:
                    
        """)

# Main content
st.title("Titanic Survival Prediction")
st.markdown("Predict whether a passenger would have survived the Titanic disaster")

with st.form("prediction_form"):
    # Passenger Details
    st.header("Passenger Information")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Class", options=list(MAPPINGS['class'].keys()), key='pclass')
        st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, 
                      step=0.5, key='age', format="%.1f")
    with col2:
        st.selectbox("Gender", options=list(MAPPINGS['gender'].keys()), key='gender')
        st.selectbox("Embarked", options=list(MAPPINGS['embarked'].keys()), key='embarked')
    
    # Title and Cabin
    st.selectbox("Title", options=list(MAPPINGS['title'].keys()), key='title')
    st.checkbox("Had Cabin", value=False, key='had_cabin')

    # Family Information
    st.header("Family Details")
    col3, col4 = st.columns(2)
    with col3:
        st.number_input("Siblings/Spouses", min_value=0, max_value=10, 
                       value=0, key='sibsp')
    with col4:
        st.number_input("Parents/Children", min_value=0, max_value=10, 
                       value=0, key='parch')

    # Fare Information
    st.header("Ticket Information")
    st.number_input("Fare ", min_value=0.0, max_value=600.0, value=32.0, 
                   step=1.0, key='fare', format="%.2f")

    # Submit button
    submitted = st.form_submit_button("Predict Survival", 
                                    disabled=not api_healthy,
                                    help="API must be available to predict")

# Handle form submission
if submitted:
    try:
        input_data = get_input_data()
        start_time = datetime.now()
        
        with st.spinner("Analyzing passenger data..."):
            response = requests.post(
                PREDICT_URL,
                json=input_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response_time = (datetime.now() - start_time).total_seconds()
            
        if response.status_code == 200:
            result = response.json()
            st.divider()
            
            # Main result display
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if result['prediction'] == 1:
                    st.success("##  Survived")
                    st.balloons()
                else:
                    st.error("## Did Not Survive")
                
            with col_res2:
                prob = result['probability'] if result['prediction'] == 1 else 1 - result['probability']
                st.metric("Confidence", f"{prob:.1%}")
                st.caption(f"Response time: {response_time:.2f}s")
            
            # Raw data and API response
            with st.expander("Technical Details"):
                st.json({
                    "input_data": input_data,
                    "api_response": result
                })
                
        else:
            st.error(f"API Error (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Documentation
with st.expander("About this app"):
    st.markdown("""
    ### Titanic Survival Predictor  
    This app predicts whether a passenger would have survived the Titanic disaster 
    based on their characteristics using a machine learning model.
    
    **Features:**
    - Real-time API health monitoring
    - Detailed passenger information collection
    - Probability-based predictions
    - Response time tracking
    
    
    """)