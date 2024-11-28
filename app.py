import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model/vanet_classifier.h5')
model.load_weights('model/vanet_classifier.weights.h5')

# Load dataset
@st.cache_data
def load_data():
    # Replace with the path to your dataset
    df = pd.read_csv('dataset\DataSet 5RoutingMetrics VANET BCN.csv')  
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select Tab", ["Explore", "Predict"])

# Tab 1: Explore
if tab == "Explore":
    st.title("Explore the Dataset")
    st.write("### Introduction")
    st.write("""
    This dataset contains traffic metrics for Vehicular Ad Hoc Networks (VANETs). 
    It is used for building predictive models to determine whether a packet will reach its destination (binary classification).
    """)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Graphs")
    st.write("#### Correlation Heatmap")
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io

    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("#### Distribution of Target Variable (OUT)")
    fig, ax = plt.subplots()
    df['OUT'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_title("Distribution of OUT")
    ax.set_xlabel("OUT")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Tab 2: Predict
elif tab == "Predict":
    st.title("Make Predictions")
    st.write("Enter values for the features to get a prediction:")

    # Input fields for user
    abw = st.number_input("Available Bandwidth (ABE):", min_value=0.0, max_value=1.0, step=0.01)
    tjr = st.number_input("Trajectory (TJR):", min_value=0.0, max_value=1.0, step=0.01)
    nv = st.number_input("Number of Neighbors (NV):", min_value=0.0, max_value=1.0, step=0.01)
    dst = st.number_input("Distance to Destination (DST):", min_value=0.0, max_value=1.0, step=0.01)
    lmac = st.number_input("MAC Losses (LMAC):", min_value=0.0)

    # Predict button
    if st.button("Predict"):
        # Prepare input data
        input_data = np.array([[abw, tjr, nv, dst, lmac]])
        prediction = model.predict(input_data)

        # Convert prediction to binary output
        result = 1 if prediction[0][0] >= 0.5 else 0
        st.write(f"### Predicted Output: {result}")
        if result == 1:
            st.success("The packet is likely to reach its destination.")
        else:
            st.error("The packet is unlikely to reach its destination.")
