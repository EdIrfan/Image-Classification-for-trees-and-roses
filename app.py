import streamlit as st
import sklearn
import joblib
import numpy as np
from PIL import Image
st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Cat and Dog classifier")
st.text("Upload the Image")

model = joblib.load("img_model")

url = input("Enter your URL")
img1 = imread(url)
if img1 is not None:
  img = Image.open(img1)
  st.image(img,caption = "Image URL")

  if st.button("Predict"):
    CATEGORIES = ["rose","tree"]
    st.write("Result..")
    flat_data=[]
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    st.write(img.shape)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.title(f"Predicted Output: {y_out}")
    q = model.predict_proba(flat_data)
    for index,item in enumerate(CATEGORIES):
      st.write(f"{item} : {q[0][index]*100}%")