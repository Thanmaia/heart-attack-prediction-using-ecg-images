# Your Streamlit app code here
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Set the image dimensions
image_width = 500
image_height = 500

# Set the path to your trained model
model_path = "C:\\Users\\THANMAI\\Downloads\\start\\model.h5"

# Load the pre-trained model
model = load_model(model_path)

# Define the labels for the classes
class_labels = ["Maximized chance of heart attack",
                "Chance of heart attack",
                "ECG of patient with heart attack",
                "Patient is healthy, no chances of heart attack"]

def main():
    st.title("ECG Heart Attack Prediction")
    st.write("Upload an ECG image and get the prediction")

    # Create a file uploader component
    uploaded_file = st.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Check if the image is not None and is a valid image file
        if image is not None and is_valid_image(image):
            image = cv2.resize(image, (image_width, image_height))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  # Normalize the image

            # Make the prediction
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)

            # Get the corresponding label
            predicted_label = class_labels[predicted_class]

            st.write("Predicted Class:", predicted_label)
        else:
            st.write("Failed to load the image. Please make sure it is a valid image file.")

def is_valid_image(image):
    # Check if the image has 3 channels (RGB/BGR) and is not grayscale
    return len(image.shape) == 3 and image.shape[2] in [3, 4]

if __name__ == '__main__':
    main()
