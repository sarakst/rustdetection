

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('nn.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Image Classification
         """
         )

map_dict = {0: 'Class A',
            1: 'Class B',
            2: 'Class C',
            3: 'Class D'}
file = st.file_uploader("Upload the image to be classified:", type=["jpg", "png","jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def upload_predict(upload_image, model):
    
        size = (64,64)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img,(64,64))
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape).argmax()
        #pred_class=decode_predictions(prediction,top=1)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, channels="RGB")
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        predictions = upload_predict(image, model)
        st.title("Predicted Label for the image is {}".format(map_dict [predictions]))