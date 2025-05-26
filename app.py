import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost
import sklearn


# Import the model
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()  # Stop execution if the file is not found
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop execution for any other errors

st.title("Laptop Price Predictor")

# Input fields (same as before)
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (in KG)', min_value=0.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen Size in Inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Prepare input features for prediction
    touchscreen_value = 1 if touchscreen == 'Yes' else 0
    ips_value = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size  # Calculate PPI

    # Create the query input for prediction
    query = np.array([company, type, ram, weight, touchscreen_value, 
                      ips_value, ppi, cpu, hdd, ssd, gpu, os])

    # Create a DataFrame for proper column alignment and type handling
    query_df = pd.DataFrame([query], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 
        'Ips', 'ppi', 'cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'
    ])

    # Convert categorical features to the same type as the training data if necessary
    for col in ['Company', 'TypeName', 'cpu brand', 'Gpu brand', 'os']:
        query_df[col] = query_df[col].astype(str)  # Ensure categorical features are strings

    # Print debug information
    st.write("Query DataFrame for prediction:", query_df)

    try:
        # Make prediction
        predicted_price = np.exp(pipe.predict(query_df)[0])
        # Display result
        st.success(f"The predicted price of this configuration is ₹{int(predicted_price)}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
