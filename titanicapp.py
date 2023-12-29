import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved model
loaded_model = load_model('Model_FNN_Titanic.h5')
loaded_model.load_weights('Weights_Model_FNN_Titanic.h5')

# Load the scaler
loaded_scaler = joblib.load('scaler_titanic.save')

# Function to preprocess input and make predictions
def predict_survival(input_data):
    # Preprocess the input features
    input_data = pd.DataFrame(input_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    input_data['Sex'] = input_data['Sex'].map({'male': 1, 'female': 0})
    input_data['Embarked'] = input_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    input_data.fillna(0, inplace=True)  # Fill NaN values with 0 for simplicity
    
    # Use the loaded scaler to scale the input data
    input_data_scaled = loaded_scaler.transform(input_data)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data_scaled)

    return predictions

# Streamlit app
st.title('Titanic Survival Prediction App')

# Add input elements (e.g., sliders, text input, etc.)
pclass = st.slider('Pclass', 1, 3, 2)
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, value=30)
sibsp = st.slider('SibSp', 0, 8, 0)
parch = st.slider('Parch', 0, 6, 0)
fare = st.number_input('Fare', min_value=0.0, value=30.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Get user input
user_input = {'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}

# Add a button to trigger the prediction
if st.button('Predict'):
    # Preprocess input and make predictions
    predictions = predict_survival([user_input])

    # Display the predictions
    st.subheader('Prediction Results:')
    st.write(f'Probability of Survival: {predictions[0][0]:.2%}')
