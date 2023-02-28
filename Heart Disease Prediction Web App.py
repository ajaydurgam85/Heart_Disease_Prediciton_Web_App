# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:57:50 2023

@author: Ajay
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/Ajay/Desktop/ML-Project-5=(Heart Disease Prediction)/heart_train_model.sav","rb"))

def Heart_Prediction(input_data):
    
   #now loading the predictive model here
   
   #now making the input_data as numpy arrayd
   input_data_as_numpy_array = np.asarray(input_data)
   #now making the input data as numpy array reshape it for one instance
   input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

   prediction = loaded_model.predict(input_data_reshape)
   print(prediction)

   if (prediction[0]==0):
       return "The Person is Healthy"
       
   else:
       return "The Person has Heart Disease"
    

def main():
    
    st.title("Heart Disease Prediction Web App")
    

    
    age = st.text_input("Enter your age")
    sex = st.text_input("What is Your sex")
    cp = st.text_input("Enter your cp")
    trestbps= st.text_input("Enter trestbps")
    chol = st.text_input("Enter chol")
    fbs = st.text_input("Enter fbs")
    restecg=st.text_input("Enter restecg")
    thalach = st.text_input("Enter thalach")
    exang = st.text_input("Enter exang")
    oldpeak = st.text_input("Enter oldpeak")
    slope = st.text_input("Enter slope")
    ca = st.text_input("Enter ca")
    thal = st.text_input("Enter thal")
    
    disease = ''
    
    if st.button("Heart Disease Result"):
        
        disease = Heart_Prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
        st.success(disease)
        
if __name__ == '__main__':
    main()
        
       
        
        
    
    
    
    