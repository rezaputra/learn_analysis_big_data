import numpy as np
import pandas as pd
import streamlit as st
from os import path, makedirs
import os
import pickle

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def app():
	"""This application helps in running machine learning models without having to write explicit code 
	by the user. It runs some basic models and let's the user select the X and y variables. 
	"""
	
	# Load the data 
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Upload Data` page!")
	else:
		df = pd.read_csv('data/main_data.csv')
		st.dataframe(df)
		st.write(f"**Variabel yang akan diprediksi :** {df.columns[-1]}")
		st.write(f"**Variabel yang akan digunakan untuk melakukan prediksi:** \
			 {list(df.columns[:-1])}")
		
		# Label Encoding
		label_list = ["Sex","BP","Cholesterol","Na_to_K","Drug"]
		for l in label_list:
			label_encoder(df, l)

		model_path = {
			"knn" : False,
			"svm" : False,
			"rf" : False
		}

		model_dir = "data/model/"
		if path.exists(model_dir):
			knn_dir_m = model_dir+"model/"+list(model_path.keys())[0]+".sav"
			svm_dir_m = model_dir+"model/"+list(model_path.keys())[1]+".sav"
			rf_dir_m = model_dir+"model/"+list(model_path.keys())[2]+".sav"
			if path.exists(knn_dir_m):
				model_path["knn"] = knn_dir_m
			if path.exists(svm_dir_m):
				model_path["svm"] = svm_dir_m
			if path.exists(rf_dir_m):
				model_path["rf"] = rf_dir_m

		# Perform train test splits 
		st.markdown("#### Train Test Splitting")
		size = st.slider("Percentage of value division",
							min_value=0.1, 
							max_value=0.9, 
							step = 0.1, 
							value=0.8, 
							help="This is the value which will be used to divide the data for training and testing. Default = 80%")
		
		x = df.drop(["Drug"],axis=1)
		y = df.Drug

		x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=size, random_state=42, shuffle=True)

		y_train = y_train.values.reshape(-1,1)
		y_test = y_test.values.reshape(-1,1)


		st.write("Jumlah training samples:", x_train.shape[0])
		st.write("Jumlah testing samples:", x_test.shape[0])

		colb1, colb2 = st.columns([1,2])

		with colb1:
			x_var = st.radio("Pilih algoritma yang akan dipakai :",options=('KNN', 'SVM', 'Random Forest'))       

		if x_var == "KNN":
			if model_path["knn"]:
				with colb2:
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")
					if st.button(f"Retrain {x_var} ?"):
						training_knn(x_train, y_train, x_test, y_test, model_path)
			else:
				if st.button(f"Jalankan Algoritma {x_var}"):
					training_knn(x_train, y_train, x_test, y_test, model_path)
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")
		if x_var == "Random Forest":
			if model_path["rf"]:
				with colb2:
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")
					if st.button(f"Retrain {x_var} ?"):
						training_rf(x_train, y_train, x_test, y_test, model_path)
			else:
				if st.button(f"Jalankan Algoritma {x_var}"):
					training_rf(x_train, y_train, x_test, y_test, model_path)
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")
		if x_var == "SVM":
			if model_path["svm"]:
				with colb2:
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")
					if st.button(f"Retrain {x_var} ?"):
						training_svm(x_train, y_train, x_test, y_test, model_path)
			else:
				if st.button(f"Jalankan Algoritma {x_var}"):
					training_svm(x_train, y_train, x_test, y_test, model_path)
					st.markdown(f"#### Training {x_var} Selesai :white_check_mark:")


def label_encoder(df, y):
	filename = y+".sav"
	model_dir = "data/model/encoder/"
	if not path.exists(model_dir):
		makedirs(model_dir)
	if not path.exists(model_dir+filename):
		le = LabelEncoder()
		model = le.fit(df[y])
		with open(model_dir+filename, 'wb') as pickle_file:
			pickle.dump(model, pickle_file)
	else:
		with open(model_dir+filename, 'rb') as pickle_file:
			model = pickle.load(pickle_file)
	df[y] = model.transform(df[y])


def training_knn(x_train, y_train, x_test, y_test, model_path):
	with st.spinner("Proses Training..."):
		grid = {'n_neighbors':np.arange(1,120),
				'p':np.arange(1,3),
				'weights':['uniform','distance']
			}

		knn = KNeighborsClassifier(algorithm = "auto")
		knn_cv = GridSearchCV(knn,grid,cv=5)
		knn_cv.fit(x_train,y_train)
	st.write("Hyperparameters:",knn_cv.best_params_)
	st.write("Train Score:",knn_cv.best_score_)
	st.write("Test Score:",knn_cv.score(x_test,y_test))
	# Save model
	algo = "knn"
	model_path[algo] = save_model(algo, knn_cv)
	save_xy_test(x_test, y_test)


def training_rf(x_train, y_train, x_test, y_test, model_path):
	with st.spinner("Proses Training..."):
		grid = {'n_estimators':np.arange(100,1000,100),
				'criterion':['gini','entropy']
			}

		rf = RandomForestClassifier(random_state = 42)
		rf_cv = GridSearchCV(rf,grid,cv=5)
		rf_cv.fit(x_train,y_train)
	st.write("Hyperparameters:",rf_cv.best_params_)
	st.write("Train Score:",rf_cv.best_score_)
	st.write("Test Score:",rf_cv.score(x_test,y_test))
	# Save model
	algo = "rf"
	model_path[algo] = save_model(algo, rf_cv)
	save_xy_test(x_test, y_test)


def training_svm(x_train, y_train, x_test, y_test, model_path):
	with st.spinner("Proses Training..."):
		grid = {
			'C':[0.01,0.1,1,10],
			'kernel' : ["linear","poly","rbf","sigmoid"],
			'degree' : [1,3,5,7],
			'gamma' : [0.01,1]
		}

		svm  = SVC ()
		svm_cv = GridSearchCV(svm, grid, cv = 5)
		svm_cv.fit(x_train,y_train)
	st.write("Hyperparameters:",svm_cv.best_params_)
	st.write("Train Score:",svm_cv.best_score_)
	st.write("Test Score:",svm_cv.score(x_test,y_test))
	# Save model
	algo = "svm"
	model_path[algo] = save_model(algo, svm_cv)
	save_xy_test(x_test, y_test)


def save_model(algo, model):
	filename = algo+".sav"
	model_dir = "data/model/model/"
	if not path.exists(model_dir):
		makedirs(model_dir)
	with open(model_dir+filename, 'wb') as pickle_file:
		pickle.dump(model, pickle_file)
	return model_dir+filename


def save_xy_test(x_test, y_test):
	model_dir_enc = "data/model/xy_test/"
	if not path.exists(model_dir_enc):
		makedirs(model_dir_enc)
	with open(model_dir_enc+"x_test.npy", "wb") as f:
		np.save(f, x_test)
	with open(model_dir_enc+"y_test.npy", "wb") as f:
		np.save(f, y_test)