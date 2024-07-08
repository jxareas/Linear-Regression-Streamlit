import streamlit as st
from utils import lottie
from streamlit_lottie import st_lottie

lottie_robot_animation = lottie.load_url("https://assets9.lottiefiles.com/packages/lf20_xaxycw1s.json")

st.set_page_config(
    page_title="Modelos Lineales",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Modelos Lineales")

st.header("Bienvenido!")
