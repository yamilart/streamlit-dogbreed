from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import os
import requests
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
from PIL import Image
from io import BytesIO
from scipy import ndimage, misc


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

if dog_image:
    if dog_image is not None:
        dogimg = Image.open(dog_image).resize(size=(224, 224))
        st.image(dog_image, width = 300)
        st.write("")
        st.write("Classifying...")
        dogimg = dog_image.read()
        dogimg = np.array(dogimg)
        result_prob = model.predict(dogimg)
        result = result_prob.argmax(axis=-1)
        result = labenc.inverse_transform(result)
        #x = image.img_to_array(dogimg)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #result_prob = model.predict(x)
        #result = result_prob.argmax(axis=-1)
        #result = labenc.inverse_transform(result)
        st.title("I'm " + str(float(round(np.amax(result_prob)*100,2))) + '% sure this cute dog is a ' + result[0])
