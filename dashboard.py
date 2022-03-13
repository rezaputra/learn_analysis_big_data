import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from statistics import mean, median, stdev, mode
import sys
import warnings
warnings.filterwarnings("ignore")

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

df = pd.read_csv("dataset/drug200.csv")

st.markdown("## Sekilas struktur Dataset")
st.table(df.head())

cola1, cola2 = st.columns(2)

with cola1:
	st.markdown("##### Statistik deskriptif")
	st.table(df.describe())

with cola2:
	st.markdown("##### Tipe data dari Dataset")
	with st_stdout("text"):
		print(df.info())

st.markdown("## Univariate Variable Analysis")

var_analysis_option = st.selectbox(
	'Pilih Variabel',
	df.columns)

# st.text(type(var_analysis_option))

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

st.markdown("## Basic Data Analysis")

var_multi_options = st.multiselect(
    'Pilih pasangan independent variable beserta dependent variable-nya',
    # ['Green', 'Yellow', 'Red', 'Blue'],
    df.columns,
    default=[df.columns[0],df.columns[-1]])

if len(var_multi_options) == 2:
	valX, valY = "", ""
	if var_multi_options[0] == df.columns[-1]:
		valX = var_multi_options[0]
		valY = var_multi_options[1]
	elif var_multi_options[1] == df.columns[-1]:
		valX = var_multi_options[1]
		valY = var_multi_options[0]
	# st.text(valX)
	if valX:
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
else:
	pass