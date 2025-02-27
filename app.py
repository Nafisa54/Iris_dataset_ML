import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

# Load the trained model

with open('first_iris_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# STREAMLIT UI
st.title('Iris Flower Prediction App')
st.write('This app predicts the **Iris Flower** Type!')
st.write('Please input the values below to get the prediction')

# Input form
sepal_ID =st.number_input('Sepal_ID', min_value=0.0, max_value=10.0, value=5.0)
sepal_length =st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button('Predict'):
    user_input = np.array([sepal_ID, sepal_width, petal_length, petal_width]).reshape(1, -1)
    prediction = model.predict(user_input)
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    st.write(f'The predicted species is: {species_mapping[int(prediction[0])]}')

#footer
st.write('Made with ❤️ Streamlit)\n')

# Run the app
