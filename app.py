import streamlit as st
import sklearn
import joblib
import numpy as np
from PIL import Image
st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Cat and Dog classifier")
st.text("Upload the Image")

model = joblib.load("https://drive.google.com/file/d/1brh3wISUqakqd5lftlo7Dgqay7R9iQl8/view?usp=sharing")

uploaded_file = st.file_uploader("Choose an image..",type="jpg")
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption = "Uploaded Image")

  if st.button("Predict"):
    CATEGORIES = ["cat","dog"]
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