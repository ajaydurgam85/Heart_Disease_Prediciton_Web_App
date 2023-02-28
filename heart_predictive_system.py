# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:54:57 2023

@author: Ajay
"""

import numpy as np
import pickle

loaded_model = pickle.load(open("C:/Users/Ajay/Desktop/ML-Project-5=(Heart Disease Prediction)/heart_train_model.sav","rb"))

#now loading the predictive model here
input_data = (58,1,2,112,230,0,0,165,0,2.5,1,1,3)
#now making the input_data as numpy arrayd
input_data_as_numpy_array = np.asarray(input_data)
#now making the input data as numpy array reshape it for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0]==0):
    print("The Person is Healthy")
    
else:
    print("The Person has Heart Disease")