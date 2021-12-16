from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from keras.models import load_model


st.set_page_config(
    page_title="Who let the dogs out?",
    page_icon="🐾"
)

model = load_model('-20-breeds.h5')

breedselection = ['dachshund', 'golden_retriever', 'chow', 'siberian_husky', 
                                                'great_dane', 'french_bulldog', 'rottweiler', 'cocker_spaniel', 
                                                'pekinese', 'doberman', 'boxer', 'labrador_retriever', 
                                                'samoyed', 'beagle', 'chihuahua', 'toy_terrier', 
                                                'weimaraner', 'collie', 'bloodhound', 'yorkshire_terrier']

st.title("Who let the dogs out?")
st.markdown("A dog breed detection project")

dog_image = st.file_uploader('Add a cute dog here! ⬇', type=['jpg'])
submit = st.button('Guess the breed')

if submit:
    if dog_image is not None:
        img = dog_image
        st.write('## Your Image')
        st.image(img, width=200)
        
        file_bytes = np.asarray(dog_image.read(), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels='BGR')
        opencv_image = cv2.resize(opencv_image, (224, 224))

        opencv_image.shape = (1, 224, 224, 3)

        Y_pred = model.predict(opencv_image)

        st.title('Prediction', breedselection[np.argmax(Y_pred)])
