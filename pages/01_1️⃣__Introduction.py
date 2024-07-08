import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import lottie
from streamlit_lottie import st_lottie

from utils.streamlit_utils import br, get_dataframe

lottie_robot_animation = lottie.load_url("https://assets9.lottiefiles.com/packages/lf20_xaxycw1s.json")

st.title("Modelos Lineales")

st.header("Bienvenido!")

st.markdown(
    "Esta simple aplicacion de Streamlit presenta un Modelo Lineal con Regularizacion L2 con el objetivo de predecir "
    "el valor de casas."
    " utilizando **Metodos Numericos** (Ecuacion Normal Regularizada), con el lenguaje Python.")
st.markdown("Feel free to check the code, learn or point out any issues!")

st_lottie(lottie_robot_animation, height=200, quality="high")

br()
st.subheader("Data Preview: ")

url = "https://raw.githubusercontent.com/jxareas/ml-zoomcamp-2022/master/02-regression/data/housing.feather"
df = pd.read_feather(url)
st.dataframe(df.head(n=10))
br()


# Define the functions from your script
def train_linear_regression(X, y):
    ones = np.ones(len(X))
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    betas = XTX_inv.dot(X.T).dot(y)
    return betas


def predict(X, betas):
    return betas[0] + X.dot(betas[1:])


# Feature selection
df['median_house_value_log'] = np.log1p(df['median_house_value'])
features = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
            'median_income']
target = 'median_house_value_log'

# Prepare the data
X = df[features]
y = df[target]

# Handle missing values by replacing them with the mean of the column
total_bedrooms_mean = X['total_bedrooms'].mean()
X['total_bedrooms'].fillna(total_bedrooms_mean, inplace=True)

# Train the model
betas = train_linear_regression(X.values, y.values)

# Create a Streamlit app
st.title("Housing Price Prediction")

st.write("""
Enter the feature values to get a prediction of the log-transformed median house value.
""")

# Input fields for feature values
latitude = st.number_input('Latitude', value=df['latitude'].min())
longitude = st.number_input('Longitude', value=df['longitude'].min())
housing_median_age = st.number_input('Housing Median Age', value=df['housing_median_age'].min())
total_rooms = st.number_input('Total Rooms', value=df['total_rooms'].min())
total_bedrooms = st.number_input('Total Bedrooms', value=total_bedrooms_mean)
population = st.number_input('Population', value=df['population'].min())
households = st.number_input('Households', value=df['households'].min())
median_income = st.number_input('Median Income', value=df['median_income'].min())

# Predict button
if st.button("Predict"):
    input_features = np.array(
        [[latitude, longitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
    prediction_log = predict(input_features, betas)
    prediction = np.expm1(prediction_log)  # Inverse of the log transformation
    st.write(f"Predicted Median House Value: ${prediction[0]:,.2f}")

    # Plotting the prediction result: input features and the predicted value
    st.subheader("Input Features and Predicted Value")

    feature_names = ['Latitude', 'Longitude', 'Housing Median Age', 'Total Rooms',
                     'Total Bedrooms', 'Population', 'Households', 'Median Income']
    feature_values = [latitude, longitude, housing_median_age, total_rooms,
                      total_bedrooms, population, households, median_income]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the feature values
    ax.barh(feature_names, feature_values, color='blue')

    # Plotting the predicted value as a line
    ax.axvline(x=prediction[0], color='red', linestyle='--', label=f'Predicted Value: ${prediction[0]:,.2f}')

    ax.set_xlabel('Feature Values and Predicted Value')
    ax.set_title('Input Features and Predicted Median House Value')
    ax.legend()

    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    pass
