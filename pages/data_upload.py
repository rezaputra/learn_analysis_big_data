import streamlit as st
import numpy as np
import pandas as pd
from pages import utils

# @st.cache
def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    global data, data_exist
    # st.text(uploaded_file)
    # data = pd.read_csv(uploaded_file)
    # st.table(data)
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data_exist = True
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    else:
        data_exist = False
    
    
    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        if data_exist:
            # Raw data 
            st.dataframe(data)
            #utils.getProfile(data)
            #st.markdown("<a href='output.html' download target='_blank' > Download profiling report </a>",unsafe_allow_html=True)
            #HtmlFile = open("data/output.html", 'r', encoding='utf-8')
            #source_code = HtmlFile.read() 
            #components.iframe("data/output.html")# Save the data to a new file 
            st.write("data.to_csv('data/main_data.csv', index=False)")
            data.to_csv('data/main_data.csv', index=False)
            
            #Generate a pandas profiling report
            #if st.button("Generate an analysis report"):
            #    utils.getProfile(data)
                #Open HTML file

            # 	pass

            # Collect the categorical and numerical columns 
            
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
            
            # Save the columns as a dataframe or dictionary
            columns = []

            # Iterate through the numerical and categorical columns and save in columns 
            columns = utils.genMetaData(data) 
            
            # Save the columns as a dataframe with categories
            # Here column_name is the name of the field and the type is whether it's numerical or categorical
            columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
            columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)

            # Display columns 
            st.markdown("**Column Name**-**Type**")
            for i in range(columns_df.shape[0]):
                st.write(f"{i+1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")