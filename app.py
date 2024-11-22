
# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle


# Load the instances that were created

with open('KMeans_model.pkl','rb') as file:
    model = pickle.load(file)

with open('pcal_final.pkl','rb') as file:
    pca = pickle.load(file)

with open('Scaler.pkl','rb') as file:
    sc = pickle.load(file)

# Define a function for prediction
def prediction(input_data):

    #Scale the data
    scaled_data = sc.transform(input_data)

    #PCA data
    pca_data = pca.transform(scaled_data)

    #Getting Predictions
    pred = model.predict(pca_data)[0]


    # Return the labels accordingly
    if pred==0:
        return 'Developing'
    elif pred==1:
        return 'Under Developed'
    else:
        return 'Developed'


# Define a function to run the application
def main():

    # For the title of the page
    st.title('HELP International Foundation')

    # Sub Text og the page
    st.subheader('This application will give the status of the country based on the socio-economic factors')

    # create a box for child mortality rate
    ch_mort = st.text_input('Enter the child mortality rate:')

    # create a box for export
    exp = st.text_input('Enter Exports (% GDP):')

    # create a box for health
    health = st.text_input('Enter Expenditure on Health (% GDP):')

    # create a box for import
    imp = st.text_input('Enter Imports (% GDP):')

    # create a box for income
    income = st.text_input('Enter average income:')

    # create a box for inflation
    infla = st.text_input('Enter Inflation:')

    # create a box for Life Expectancy
    life = st.text_input('Enter Life Expectancy:')

    # create a box for Fertility rate
    fert = st.text_input('Enter Fertility rate:')
    
    # create a box for GDP
    gdp = st.text_input('Enter GDP per Population:')

    # Save all the input in a 2-D list
    input_list = [[ch_mort,exp,health,imp,income,infla,life,fert,gdp]]

    # Create a button to predict
    if st.button("Predict"):
        response = prediction(input_list)
        st.success(response)

# To execute the main function
if  __name__ == '__main__':
    main()
