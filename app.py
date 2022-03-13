import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload, univariate_data_visualize, machine_learning, \
	 compare_model, drug_prediction # import your pages here

# Create an instance of the app 
app = MultiPage()

st.title("Aplikasi Rekomendasi Jenis Obat")

app.add_page("Upload Data", data_upload.app)
app.add_page("Basic Data Analysis", univariate_data_visualize.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Model Comparison", compare_model.app)
app.add_page("Drug Prediction", drug_prediction.app)

# The main app
app.run()
