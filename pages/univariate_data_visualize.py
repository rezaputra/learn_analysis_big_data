import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statistics import mean, median, stdev, mode

def app():
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Upload Data` page!")
	else:
		df_analysis = pd.read_csv('data/main_data.csv')

		st.markdown("### Univariate Variable Analysis")

		var_analysis_option = st.selectbox(
			'Pilih Variabel',
			df_analysis.columns)
		
		univariate_variable_analysis(df_analysis, var_analysis_option)

		st.markdown("### Basic Data Analysis")
		col1, col2 = st.columns(2)

		x_var = col1.radio("Pilih variabel x", options=df_analysis.columns)       

		with col2:
			y_var = st.selectbox(
				'Pilih Variabel y (label)',
				df_analysis.columns)
		
		if x_var and x_var != df_analysis.columns[-1]:
			if y_var == df_analysis.columns[-1]:
				# st.text(x_var, )
				# st.text(type(x_var))
				# st.text(y_var, )
				# st.text(type(y_var))
				xy_data_analysis(df_analysis, y_var, x_var)
		

def univariate_variable_analysis(df, var_analysis_option):

	colb1, colb2 = st.columns([3,1])

	with colb1:
		# Age distribution
		plt.figure(figsize = (9,5))
		if ((df.columns[0] == var_analysis_option) or 
				(df.columns[4] == var_analysis_option)):
			sns.distplot(df[var_analysis_option])
		elif ((df.columns[1] == var_analysis_option) or 
				(df.columns[2] == var_analysis_option) or
				(df.columns[3] == var_analysis_option) or
				(df.columns[5] == var_analysis_option)):
			sns.countplot(df[var_analysis_option])
		st.pyplot(plt)

	with colb2:
		if ((df.columns[0] == var_analysis_option) or 
				(df.columns[4] == var_analysis_option)):
			# st.text(mean(df[var_analysis_option]))
			var_analysis_option_df = pd.Series(
				[df[var_analysis_option].max(), df[var_analysis_option].min(),
				median(df[var_analysis_option]), mode(df[var_analysis_option]),
				mean(df[var_analysis_option]), stdev(df[var_analysis_option])],
				index=['max', 'min', 'median', 'mode', 'mean', 'stdev'])
			st.dataframe(var_analysis_option_df)
		elif ((df.columns[1] == var_analysis_option) or 
				(df.columns[2] == var_analysis_option) or
				(df.columns[3] == var_analysis_option) or
				(df.columns[5] == var_analysis_option)):
			st.table(df[var_analysis_option].value_counts())


def xy_data_analysis(df, valX, valY):
	if ((df.columns[0] == valY) or 
			(df.columns[4] == valY)):
		plt.figure(figsize=(9,5))
		sns.swarmplot(x = valX, y = valY, data = df)
		plt.legend(df[valX].value_counts().index)
		plt.title(f"{valY} -- {valX}")
		st.pyplot(plt)
	elif ((df.columns[1] == valY) or 
			(df.columns[2] == valY) or
			(df.columns[3] == valY) or
			(df.columns[5] == valY)):
		df_combine = df.groupby([valX, valY]).size().reset_index(name = "Count")
		plt.figure(figsize = (9,5))
		sns.barplot(x = valX,y= "Count", hue = valY,data = df_combine)
		plt.title(f"{valY} -- {valX}")
		st.pyplot(plt)