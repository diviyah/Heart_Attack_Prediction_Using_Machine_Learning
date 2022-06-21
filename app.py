# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:30:36 2022

@author: ASUS
"""

import pickle
import os
import numpy as np
import streamlit as st

# Can use either best_pipeline_model or best_GSCV_model
# Because both the models are producing the same accuracy when tested with test dataset
# which is 80.22%

BEST_PIPE_PATH = os.path.join(os.getcwd(),'model','best_pipeline_model.pkl')

with open(BEST_PIPE_PATH,'rb') as file:
    model = pickle.load(file)
    
    
# Selected features from developed model
# X_new = ['age','trtbps','chol','thalachh','oldpeak','cp','thall']

# Simply took out the first row of data from X to play around
X_new = [63,145,233,150,2.3,3,1]


outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
print(outcome)

with st.form("my_form"):
    age = st.number_input("age")
    trtbps = st.number_input("trtbps")
    chol = st.number_input("chol")
    thalachh = st.number_input("thalachh")
    oldpeak = st.number_input("oldpeak")
    cp = st.number_input("cp")
    thall = st.number_input("thall")
       
        
    # Every form must have a submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,trtbps,chol,thalachh,oldpeak,cp,thall]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        st.write(outcome)