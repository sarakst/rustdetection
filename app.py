

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
         # Rust Detection
         """
         )

map_dict = {0: 'Class A',
            1: 'Class B',
            2: 'Class C',
            3: 'Class D'}
images = st.file_uploader("Upload the image to be classified:", type=["jpg", "png","jpeg"],
                        accept_multiple_files= True)
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
if images is None:
    st.text("Please upload an image file")
else:
    Genrate_pred = st.button("Generate Prediction")
    col1,col2,col3,col4  = st.columns(4)
    with col1:
        st.header("Class A")
    with col2:
        st.header("Class B")
    with col3:
        st.header("Class C")
    with col4:
        st.header("Class D")
    for file in images:
        image = Image.open(file)
        #st.image(image, channels="RGB")
        if Genrate_pred:
            predictions = upload_predict(image, model)
            #st.title("Predicted Label is {}".format(map_dict [predictions]))
            if predictions == 0:
                col1.image(image, channels="RGB")
            if predictions == 1:
                col2.image(image, channels="RGB")
            if predictions == 2:
                col3.image(image, channels="RGB")
            if predictions == 3:
                col4.image(image, channels="RGB")