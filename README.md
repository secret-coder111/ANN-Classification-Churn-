# Bank Customer Churn Prediction

This project is an end-to-fully-fledged Artificial Neural Network (ANN) application that predicts customer churn in a banking context. It uses an ANN built with TensorFlow and Keras, alongside a modern web dashboard built with Streamlit.

## Features
- **Deep Learning Model (ANN)**: Trained using TensorFlow/Keras to predict churn probabilities. Handles class imbalance and prevents overfitting via Dropout layers.
- **Data Preprocessing**: Comprehensive handling of numerical scaling (`StandardScaler`), One-Hot Encoding for geography, and Label Encoding for gender.
- **Interactive UI**: A rich web dashboard made with Streamlit and Plotly for beautiful gauge visualizations, tracking customer risk metrics and providing retention recommendations.
- **Easy Check**: Input new customer details through an interactive sidebar to see predictions instantly.

## Technologies Used
- Python 3.x
- TensorFlow & Keras
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Plotly

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd <your-repo-directory>
   ```

2. **Install the dependencies:**
   Make sure you have an environment active:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional)**:
   You can run `experiments.ipynb` to retrain the model and generate new artifact files (`model.h5`, `scaler.pkl`, `onehotencoder.pkl`, `label_encoder_gender.pkl`).

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Previews
*(Add a screenshot of your Streamlit app here!)*

## License
MIT License
