# Import library yang dibutuhkan
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load model yang sudah dilatih
model = load_model('model3.h5')

# Fungsi untuk memprediksi gambar
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

# Streamlit App
def main():
    st.title("Pizza Classification App")
    st.write("Upload an image to classify whether it's a pizza or not.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        prediction = predict_image(uploaded_file, model)

        # Display the result
        st.write("Prediction Probability:", prediction[0][0])
        if prediction[0][0] > 0.5:
            st.write("It's a Pizza!")
        else:
            st.write("It's not a Pizza!")

if __name__ == "__main__":
    main()
