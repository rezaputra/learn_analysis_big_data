import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle

def app():
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Upload Data` page!")
	else:
		df = pd.read_csv('data/main_data.csv')

		model_dir = "data/model/"

		model_path = {
			"knn" : False,
			"svm" : False,
			"rf" : False
		}

		loaded_model = {
			"knn" : False,
			"svm" : False,
			"rf" : False
		}

		if os.path.exists(model_dir):
			knn_dir_m = model_dir+"model/"+list(model_path.keys())[0]+".sav"
			svm_dir_m = model_dir+"model/"+list(model_path.keys())[1]+".sav"
			rf_dir_m = model_dir+"model/"+list(model_path.keys())[2]+".sav"
			if os.path.exists(knn_dir_m):
				model_path["knn"] = knn_dir_m
				with open(model_path["knn"], 'rb') as pickle_file:
					loaded_model["knn"] = pickle.load(pickle_file)
			if os.path.exists(svm_dir_m):
				model_path["svm"] = svm_dir_m
				with open(model_path["svm"], 'rb') as pickle_file:
					loaded_model["svm"] = pickle.load(pickle_file)
			if os.path.exists(rf_dir_m):
				model_path["rf"] = rf_dir_m
				with open(model_path["rf"], 'rb') as pickle_file:
					loaded_model["rf"] = pickle.load(pickle_file)

		cola1, cola2 = st.columns([1,3])

		with cola1:
			val_sex = st.radio(label="Jenis Kelamin:", options=('MALE', "FEMALE"))
			val_chl = st.radio(label="Cholesterol Levels", options=('HIGH',
															'NORMAL'))
			val_bp = st.radio(label="Blood Pressure Levels",
								options=('HIGH', 'LOW', 'NORMAL'))
		
		if val_sex == "MALE":
			val_sex = "M"
		else:
			val_sex = "F"

		track_options_model = {}

		with cola2:
			val_age = st.slider(label="Usia", min_value=15, max_value=80)
			val_age = int(val_age)
			val_nak = st.slider(label="Na to Potassium Ration", min_value=6.0, max_value=39.0)
			val_nak = float(val_nak)
			options = []
			if os.path.exists(model_dir+"model"):
				l_models = os.listdir(model_dir+"model")
				for l_model in l_models:
					if l_model[:-4] == "knn":
						options.append("KNN")
						track_options_model["KNN"] = loaded_model["knn"]
					if l_model[:-4] == "svm":
						options.append("SVM")
						track_options_model["SVM"] = loaded_model["svm"]
					if l_model[:-4] == "rf":
						options.append("Random Forest")
						track_options_model["Random Forest"] = loaded_model["rf"]
			if len(options) >= 1:
				select_model = st.selectbox(label="Pilih Model", options=tuple(options))
			else:
				st.write("Model Belum Tersedia")
		
		transform_label_enc = np.array([[val_age, val_sex, val_bp, val_chl, val_nak]])
		transform_label_enc = pd.DataFrame(transform_label_enc, columns=df.columns[:-1])

		for i in df.columns[1:-2]:
			with open(model_dir+"encoder/"+i+".sav", 'rb') as pickle_file:
				model_enc = pickle.load(pickle_file)
			transform_label_enc[i] = model_enc.transform(transform_label_enc[i])

		colb1, colb2 = st.columns([1,3])
		btn_invoke = False
		with colb2:
			if st.button("Prediksi Jenis Obat"):
				btn_invoke = True
		if btn_invoke:
			predicted_result = track_options_model[select_model].predict(transform_label_enc)
			with open(model_dir+"encoder/"+df.columns[-1]+".sav", 'rb') as pickle_file:
				model_enc_y = pickle.load(pickle_file)
			predicted_result = model_enc_y.inverse_transform(predicted_result)
			st.success(f"Obat yang direkomendasikan adalah :pill: {predicted_result[0]} :pill:")
