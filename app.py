import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.block-container { padding-top: 1.5rem; }
.metric-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0px 2px 8px rgba(0,0,0,0.15); }
.title { text-align:center; color:#1E3A8A; margin-bottom: 20px; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; padding: 10px 16px; font-size: 16px;}
.stTabs [aria-selected="true"] { background-color: #e2e8f0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL (OPTIMIZED & CACHED)
# ==========================================
@st.cache_resource
def load_artifacts():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    model = tf.keras.models.load_model("model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("onehotencoder.pkl", "rb") as f:
        onehotencoder = pickle.load(f)
    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    return model, scaler, onehotencoder, label_encoder

def process_and_predict(model, scaler, onehotencoder, label_encoder, input_dict):
    input_data = pd.DataFrame([input_dict])
    input_data['Gender'] = label_encoder.transform(input_data['Gender'])
    
    geo_encoded = onehotencoder.transform(input_data[['Geography']]).toarray()
    geo_columns = onehotencoder.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)
    
    input_data = input_data.drop('Geography', axis=1)
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)
    
    feature_names = input_data.columns.tolist()
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    
    return float(prediction[0][0]), input_scaled[0], feature_names

def main():
    with st.spinner('Loading Deep Learning Models...'):
        model, scaler, onehotencoder, label_encoder = load_artifacts()

    # ==========================================
    # HEADER & TABS
    # ==========================================
    st.markdown("<h1 class='title'>📈 Intelligent Churn Radar</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction Dashboard", "📊 Profile Data", "🧠 Model Info"])

    # ==========================================
    # SIDEBAR INPUTS (WITH TOOLTIPS)
    # ==========================================
    with st.sidebar:
        st.header("👤 Customer Inputs")
        st.info("Input the demographics and financial data of the customer below.")
        
        geography = st.selectbox("🌍 Geography", onehotencoder.categories_[0], help="Country of residence")
        gender = st.selectbox("🚻 Gender", label_encoder.classes_)
        age = st.slider("🎂 Age", 18, 100, 35, help="Older customers tend to have different financial stability.")
        credit_score = st.number_input("💳 Credit Score", 300, 900, 650, help="A higher score usually indicates better financial health.")
        balance = st.number_input("💰 Account Balance ($)", min_value=0.0, value=50000.0)
        estimated_salary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, value=100000.0)
        tenure = st.slider("🗓️ Tenure (Years)", 0, 10, 5, help="How many years the customer has been with the bank.")
        num_of_products = st.slider("📦 Number Of Products", 1, 4, 2, help="Number of bank products (e.g. loans, credit cards, saving accounts).")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            has_cr_card = st.selectbox("Credit Card?", [0, 1], help="1 = Yes, 0 = No")
        with col_c2:
            is_active_member = st.selectbox("Active?", [0, 1], help="1 = Active last month, 0 = Inactive")

        predict_btn = st.button("🚀 Analyze Churn Risk", type="primary", use_container_width=True)

    if 'probability' not in st.session_state:
        st.session_state.probability = None

    # ==========================================
    # PREDICTION LOGIC
    # ==========================================
    if predict_btn:
        input_dict = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }

        with st.spinner("Neural Network is processing..."):
            prob, scaled_features, feature_names = process_and_predict(model, scaler, onehotencoder, label_encoder, input_dict)
            st.session_state.probability = 1-prob
            st.session_state.retention_probability = prob
            st.session_state.input_dict = input_dict
            st.session_state.scaled_features = scaled_features
            st.session_state.feature_names = feature_names

    # ==========================================
    # TAB 1: PREDICTION DASHBOARD
    # ==========================================
    with tab1:
        if st.session_state.probability is not None:
            prob = st.session_state.probability
            retention = st.session_state.retention_probability

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="⚠️ Churn Probability", value=f"{prob*100:.2f}%")
            with col2:
                st.metric(label="🛡️ Retention Probability", value=f"{retention*100:.2f}%")

            st.write("### Churn Risk Meter")
            st.progress(prob)

            if prob < 0.30:
                st.success("🟢 **Low Risk Customer** - Customer is likely to stay.")
            elif prob < 0.70:
                st.warning("🟡 **Medium Risk Customer** - Customer shows some signs of dissatisfaction.")
            else:
                st.error(f"🔴 **High Risk Customer** - Customer is likely to churn. (Probability: {prob*100:.2f}%)")
                st.subheader("Recommended Retention Actions")
                st.markdown("""
                - Offer loyalty rewards
                - Provide personalized offers
                - Contact customer proactively
                - Improve engagement programs
                - Offer reduced fees
                """)

        else:
            st.info("👈 Please enter customer details in the sidebar and click **Analyze Churn Risk** to see results.")

    # ==========================================
    # TAB 2: PROFILE DATA
    # ==========================================
    with tab2:
        if st.session_state.probability is not None:
            st.subheader("Customer Summary")
            
            # Formatting the dictionary into a nicer vertical dataframe
            summary_df = pd.DataFrame([st.session_state.input_dict]).T
            summary_df.columns = ["Value"]
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("Deviation Profile")
            st.write("This shows how far the customer is from the statistical average (0.0).")
            dev_df = pd.DataFrame({
                "Feature": st.session_state.feature_names,
                "Deviation (Standard Deviations)": st.session_state.scaled_features
            }).set_index("Feature")
            
            # Using native Streamlit bar chart rather than Plotly
            st.bar_chart(dev_df)
        else:
            st.warning("No data analyzed yet. Run a prediction first.")

    # ==========================================
    # TAB 3: MODEL INFO
    # ==========================================
    with tab3:
        st.subheader("🧠 Deep Learning Architecture")
        st.write("This dashboard runs a custom Artificial Neural Network built with TensorFlow and Keras.")
        if st.session_state.probability is not None:
            confidence = max(st.session_state.probability, st.session_state.retention_probability)
            st.info(f"Current Prediction Confidence : **{confidence*100:.2f}%**")
        st.markdown("""
        * **Input Layer:** Standardized financial and demographic features
        * **Hidden Layers:** Multi-layer Dense network with ReLU activations
        * **Output Layer:** Single neuron with Sigmoid activation function
        """)

if __name__ == "__main__":
    main()