import streamlit as st
import numpy as np
import cv2
from predict import class_dict, predict_4class, predict_2class
import time

st.title('X-Ray Disease Classification')

image_file = st.file_uploader("Upload a chest X-Ray image", type=['png', 'jpg'])

if image_file is not None:
    start_time = time.time() # To calculate time taken

    image_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    st.write('Input Image:')
    #st.write('Image size: ' + str(image.shape))
    st.image(image)

    predict = predict_4class(image)

    st.write('Prediction : ' + class_dict[predict])

    time_taken = round(time.time() - start_time,2)
    st.write("Time taken: " + str(time_taken) + " seconds")
