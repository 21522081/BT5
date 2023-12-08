import streamlit as st
from PIL import Image 
import pickle as pkl
import numpy as np

class_list = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9':'9','10':'10'}

st.title('Ảnh chữ số viết tay')

input = open('lrc_mnist.pkl','rb')
model = pkl.load(input)

st.header('Upload an image')
image = st.file_uploader('Choose an image', type=(['png', 'jpg', 'jpeg']))

if image is not None:
  image = Image.open(image)
  st.image(image, caption='Test image')

  if st.button('Predict'):
    image = image.resize((8*8,1))
    vector = np.array(image)
    label= str(st.write(model.predict(vector))[0])

    st.header('Result')
    st.text(class_list[label])
