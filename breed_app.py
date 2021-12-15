from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import os
import cv2
import numpy as np


REPO_DIR = 'https://github.com/willjobs/dog-classifier/raw/main'
MODEL_FILE = '-20-breeds.h5'

def load_model(model_path):
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer":hub.KerasLayer})
  return model

st.set_page_config(
    page_title="Who let the dogs out?",
    page_icon="🐾"
)

st.title("Who let the dogs out?")
st.markdown("A dog breed detection project")

file_data = st.file_uploader("Select an image", type=["jpg"])


def download_file(url):
    with st.spinner('Downloading model...'):
        # from https://stackoverflow.com/a/16696317
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
        return local_filename

def fix_rotation(file_data):
    # check EXIF data to see if has rotation data from iOS. If so, fix it.
    try:
        image = PILImage.create(file_data)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif = dict(image.getexif().items())

        rot = 0
        if exif[orientation] == 3:
            rot = 180
        elif exif[orientation] == 6:
            rot = 270
        elif exif[orientation] == 8:
            rot = 90

        if rot != 0:
            st.write(f"Rotating image {rot} degrees (you're probably on iOS)...")
            image = image.rotate(rot, expand=True)
            # This step is necessary because image.rotate returns a PIL.Image, not PILImage, the fastai derived class.
            image.__class__ = PILImage

    except (AttributeError, KeyError, IndexError):
        pass  # image didn't have EXIF data

    return image


# cache the model so it only gets loaded once
@st.cache(allow_output_mutation=True)
def get_model():
    if not os.path.isfile(MODEL_FILE):
        _ = download_file(f'{REPO_DIR}/models/{MODEL_FILE}')

    learn = load_learner(MODEL_FILE)
    return learn

learn = get_model()

if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = fix_rotation(file_data)
        
        st.write('## Your Image')
        st.image(img, width=200)

        # classify
        pred, pred_idx, probs = learn.predict(img)
        top5_preds = sorted(list(zip(learn.dls.vocab, list(probs.numpy()))), key=lambda x: x[1], reverse=True)[:5]

        # prepare output
        
        model = load_model(MODEL_FILE)
 
        #get the image of the dog for prediction
        dog_img_array = cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),((224,224)))
        #scale array into the range of -1 to 1.
        #expand the dimension on the axis 0 and normalize the array values
        dog_img_array = preprocess_input(np.expand_dims(np.array(dog_img_array[...,::-1].astype(np.float32)).copy(), axis=0))
 
        #feed the model with the image array for prediction
        pred_val = model.predict(np.array(dog_img_array,dtype="float32"))
 
        #display the image of dog
        #cv2.imshow(cv2.resize(cv2.imread(dog_img_path,cv2.IMREAD_COLOR),((224,224)))) 
        Image(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),((224,224)))) 
 
        #display the predicted breed of dog
        pred_breed = sorted(somedogs['breed'])[np.argmax(pred_val)]

        st.write('## This cutie is a ')
        st.markdown(pred_breed, unsafe_allow_html=True)

        st.write(f"🤔 Don't see your dog breed? For a full list of dog breeds in this project, [click here](https://htmlpreview.github.io/?https://github.com/willjobs/dog-classifier/blob/main/dog_breeds.html).")

