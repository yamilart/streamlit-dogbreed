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
    img = dog_image
    st.write('## Your Image')
    st.image(img, width=200)
    if dog_image is not None:
        image = cv2.imread(dog_image)
        image = cv2.resize(image, (224, 224))
        image = image.reshape(1,224,224,3)
        result_prob = model.predict(image)
        result = result_prob.argmax(axis=-1)
        result = labenc.inverse_transform(result)
        st.title("I'm " + str(float(round(np.amax(result_prob)*100,2))) + '% sure this cute dog is a ' + result[0])

