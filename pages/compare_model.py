import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def app():
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Upload Data` page!")
	else:
		st.markdown("#### Algoritma yang telah di-Training:")
		st.write("\n")

		# To store results of models
		result_dict_train = {}
		result_dict_test = {}

		df = pd.read_csv('data/main_data.csv')

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
		model_dir = "data/model/"

		if os.path.exists(model_dir+"xy_test"):
			l_tests = os.listdir(model_dir+"xy_test")
			with open(model_dir+"xy_test/"+l_tests[0], 'rb') as f:
				x_test = np.load(f, allow_pickle=True)
			with open(model_dir+"xy_test/"+l_tests[1], 'rb') as f:
				y_test = np.load(f, allow_pickle=True)

		if os.path.exists(model_dir):
			knn_dir_m = model_dir+"model/"+list(model_path.keys())[0]+".sav"
			svm_dir_m = model_dir+"model/"+list(model_path.keys())[1]+".sav"
			rf_dir_m = model_dir+"model/"+list(model_path.keys())[2]+".sav"
			if os.path.exists(knn_dir_m):
				model_path["knn"] = knn_dir_m
				with open(model_path["knn"], 'rb') as pickle_file:
					loaded_model["knn"] = pickle.load(pickle_file)
				result_dict_train["KNN Train Score"] = loaded_model["knn"].best_score_
				result_dict_test["KNN Test Score"] = loaded_model["knn"].score(x_test, y_test)
			if os.path.exists(svm_dir_m):
				model_path["svm"] = svm_dir_m
				with open(model_path["svm"], 'rb') as pickle_file:
					loaded_model["svm"] = pickle.load(pickle_file)
				result_dict_train["SVM Train Score"] = loaded_model["svm"].best_score_
				result_dict_test["SVM Test Score"] = loaded_model["svm"].score(x_test, y_test)
			if os.path.exists(rf_dir_m):
				model_path["rf"] = rf_dir_m
				with open(model_path["rf"], 'rb') as pickle_file:
					loaded_model["rf"] = pickle.load(pickle_file)
				result_dict_train["Random Forest Train Score"] = loaded_model["rf"].best_score_
				result_dict_test["Random Forest Test Score"] = loaded_model["rf"].score(x_test, y_test)

		options = []
		if model_path["knn"]:
			options.append('KNN')
		if model_path["svm"]:
			options.append('SVM')
		if model_path["rf"]:
			options.append('Random Forest')
		
		emots = [":guitar:", ":saxophone:", ":violin:"]

		cold1, cold2 = st.columns([1,2])

		if len(result_dict_train) == 0:
			st.markdown("##### Model Belum Tersedia :no_entry:")

		with cold1:
			for opt, emot in zip(options, emots):
				st.markdown(f"##### {emot} {opt}")
		with cold2:
			if len(result_dict_train) >= 1:
				with st.expander("Parameter Terbaik :"):
					for key, l_model in loaded_model.items():
						if l_model:
							if key == "knn":
								st.write("KNN")
							if key == "svm":
								st.write("SVM")
							if key == "rf":
								st.write("Random Forest")
							st.write(l_model.best_params_)
		
		"""Compare Algorithm Performance"""
		if (len(result_dict_train) >= 2) and (len(result_dict_test) >= 2):
			df_result_train = pd.DataFrame.from_dict(result_dict_train,
											orient="index", columns=["Train Score"])
			df_result_test = pd.DataFrame.from_dict(result_dict_test,
											orient="index", columns=["Test Score"])
			cola1, cola2, cola3 = st.columns(3)
			btn_invoke = False
			with cola2:
				if st.button("Bandingkan Performa Algoritma"):
					btn_invoke = True
			if btn_invoke:
				colb1, colb2 = st.columns(2)
				with colb1:
					st.table(df_result_train)
				with colb2:
					st.table(df_result_test)
				fig,ax = plt.subplots(1,2,figsize=(25, 10))
				sns.barplot(x = df_result_train.index,y = df_result_train["Train Score"],ax = ax[0])
				sns.barplot(x = df_result_test.index,y = df_result_test["Test Score"],ax = ax[1])
				ax[0].set_xticklabels(df_result_train.index, fontsize=19, rotation = 55)
				ax[1].set_xticklabels(df_result_test.index, fontsize=19, rotation = 55)
				st.pyplot(plt)
