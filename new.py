import pandas as pd
import numpy as np
import streamlit as st
import algo as algo



st.set_page_config(
        page_title="Surat House Price Prediction",
        layout="centered",
    )

# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
# st.markdown(hide_default_format, unsafe_allow_html=True)
text_color = 'blue'

def main():
    HP=pd.read_excel("House_Price_Surat_new.xlsx")

    # To convert the object into categorical type using the label enconding for unique value categorical
    from sklearn.preprocessing import LabelEncoder
    # Create a LabelEncoder object
    le = LabelEncoder()
    # Fit the encoder to the column you want to encode
    le.fit(HP['LOCATION'])
    # Transform the column values using the encoder
    HP['NEW_LOCATION'] = le.transform(HP['LOCATION'])


    # Same label encoding for he type as well 
    # Create a LabelEncoder object
    le2= LabelEncoder()

    # Fit the encoder to the column you want to encode
    le2.fit(HP['Type'])

    # Transform the column values using the encoder
    HP['NEW_Type'] = le2.transform(HP['Type'])


    # for data processing and filling the BHK value to 0 where plot 0 and value as 3 where bhk is Row House
    for index, row in HP.iterrows():
          if row['Type']=='Plot':
            HP.loc[index, 'BHK']=0 
          if row['Type']== 'Row House':
            HP.loc[index,'BHK']=3

    # changing the datatype to int for data processing 
    HP['BHK']= HP['BHK'].astype(int)


    
#     # Predict house prices for new data
#     new_data = pd.DataFrame([[1450, 800,2,1,12,1,69], [1250, 1100,1,0,22,0,32]], columns=['AVGRS','SQFT', 'BHK','READY_TO_MOVE','','NEW_Type','NEW_LOCATION'])
#     new_prices = dt.predict(new_data)
#     print("Predicted prices for new data:", new_prices)

# ----- UI -------

#     locations = sorted(HP['LOCATION'].unique())
#     st.text(locations)

#     type = HP['Type'].unique()
#     st.text(type)

    HP["Period"] = HP["NEW_LOCATION"].astype(str) +" - "+ HP['LOCATION']
    HP["Data"] = HP["NEW_Type"].astype(str) +" - "+ HP['Type']

#     print(HP)


    st.title(f':{text_color}[House Price Prediction]')
    
    col1, col2 = st.columns([1,1],gap='medium')
    
    with col1:
        st.text('Please Select Location')
        new_loc = st.selectbox(
            'Location',
            label_visibility='collapsed',
        options=HP["Period"].unique())
    
        st.text('Please Select Type')
        new_type = st.selectbox(
            "Algoritm",
            options=HP["Data"].unique(),
            label_visibility='collapsed')
        
        st.text('Please Select Algorithm')
        algorithm = st.selectbox(
            "Algoritm",("Linear Regression","RFR","SVM","Decisiontree"),
            label_visibility='collapsed')
       
    
    with col2:
        st.text('Please Enter bhk')
        bhk = int(st.text_input('', value=0, label_visibility='collapsed'))
    
        st.text('Please Enter SquareFoot')
        square_foot = int(st.text_input('', value=1,label_visibility='collapsed'))
    
    	
    col1, col2, col3 = st.columns([1.2,.5,1])
    placeholder = st.empty()
    
    
    new_loc=new_loc.rsplit(' - ')
    new_type=new_type.rsplit(' - ')

    
    new_data = pd.DataFrame([[square_foot,bhk,new_type[0],new_loc[0]]], columns=[ 'SQFT', 'BHK','NEW_LOCATION','NEW_Type'])
    
    with col1:
        if st.button("Predict", type='primary'):
            if algorithm == 'Linear Regression':
                with placeholder.container():
                    predicted_value = algo.lr(HP, new_data)
                    st.markdown(f"Pridicted Prices for provided data - :{text_color}{predicted_value}") 
            if algorithm == 'RFR':
                with placeholder.container():
                    predicted_value = algo.rf(HP, new_data)
                    st.markdown(f"Pridicted Prices for provided data - :{text_color}{predicted_value}") 


            if algorithm == 'SVM':
                with placeholder.container():
                    predicted_value = algo.svma(HP, new_data)
                    st.markdown(f"Pridicted Prices for provided data - :{text_color}{predicted_value}") 
            
            if algorithm == 'Decisiontree':
                with placeholder.container():
                    predicted_value = algo.dest(HP, new_data)
                    st.markdown(f"Pridicted Prices for provided data - :{text_color}{predicted_value}") 
        
    with col2:
        if st.button("Clear"):
            placeholder.empty()

    
#     st.text(new_loc)
#     st.text(new_type)
#     st.text(bhk)
#     st.text(square_foot)
#     st.text(algorithm)
    
    
    
    
    
#     ---- Pridicting ----

#     if new_loc and new_type:
        
#         temp = [new_loc, new_loc]
#         new_loc, new_loc = le.transform(temp)
    
#         temp = [new_type, new_type]
#         new_type, new_type = le2.transform(temp)
    
    
#new_data=pd.DataFrame([[ square_foot,bhk,new_type,new_loc]],columns=[ 'AVGRS','SQFT','BHK','READY_TO_MOVE','Age','NEW_Type','NEW_LOCATION'])
# #         print("Predicted prices for new data:", new_prices)
#         st.text('Pridicted Prices for provided data -')
#         st.caption(new_prices)
    

if __name__ == "__main__":
    main()
