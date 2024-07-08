import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import lottie
from streamlit_lottie import st_lottie

from utils.streamlit_utils import top_question, br, get_dataframe

lottie_robot_animation = lottie.load_url("https://assets9.lottiefiles.com/packages/lf20_xaxycw1s.json")

st.title("Modelos Lineales")

st.header("Bienvenido!")

st.markdown(
    "Esta simple aplicacion de Streamlit presenta un Modelo Lineal con Regularizacion L2 con el objetivo de predecir "
    "el valor de casas."
    " utilizando **Metodos Numericos** (Ecuacion Normal Regularizada), con el lenguaje Python.")
st.markdown("Sientete libre de echarle un vistazo al codigo o comentar cualquier defecto!")

st_lottie(lottie_robot_animation, height=200, quality="high")

br()
st.subheader("Previa de la Data: ")

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
st.title("Predicción del Precio de Viviendas")

st.write("""
Ingrese los valores de las características para obtener una predicción del valor mediano de la vivienda transformado logarítmicamente.
""")

# Input fields for feature values
latitude = st.number_input('Latitud', value=df['latitude'].min())
longitude = st.number_input('Longitud', value=df['longitude'].min())
housing_median_age = st.number_input('Edad Mediana de la Vivienda', value=df['housing_median_age'].min())
total_rooms = st.number_input('Total de Habitaciones', value=df['total_rooms'].min())
total_bedrooms = st.number_input('Total de Dormitorios', value=total_bedrooms_mean)
population = st.number_input('Población', value=df['population'].min())
households = st.number_input('Hogares', value=df['households'].min())
median_income = st.number_input('Ingreso Mediano', value=df['median_income'].min())

# Predict button
if st.button("Predecir"):
    input_features = np.array(
        [[latitude, longitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
    prediction_log = predict(input_features, betas)
    prediction = np.expm1(prediction_log)  # Inverse of the log transformation
    st.write(f"Valor del precio mediano de la vivienda: ${prediction[0]:,.2f}")

    # Plot of selected features vs prediction
    st.subheader("Selected Features vs Predicted Value")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the selected features vs prediction
    feature_names = ['Latitude', 'Longitude', 'Housing Median Age', 'Total Rooms',
                     'Total Bedrooms', 'Population', 'Households', 'Median Income']
    feature_values = input_features[0]

    # Plotting the selected features
    scatter = ax.scatter(feature_names, feature_values, label='Selected Features', color='blue', marker='o', s=100)

    # Adding tooltips with values
    for i, txt in enumerate(feature_values):
        ax.annotate(f'{txt:.2f}', (feature_names[i], feature_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Plotting the predicted value as a horizontal line
    ax.axhline(y=prediction[0], color='red', linestyle='--', label=f'Predicted Value: ${prediction[0]:,.2f}')

    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Selected Features vs Predicted Median House Value')
    ax.legend()
    ax.grid(True)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    pass
