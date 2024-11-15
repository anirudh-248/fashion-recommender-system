import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        # Ensure the uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

def save_url_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            img_path = os.path.join('uploads', 'downloaded_image.jpg')
            img.save(img_path)
            return img_path
        else:
            st.error(f"Error downloading image: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error saving image from URL: {e}")
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# File upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommendation
        indices = recommend(features, feature_list)
        # Show recommendations
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.subheader(i+1)
                st.image(filenames[indices[0][i + 1]])
else:
    # URL input
    url = st.text_input("Or enter the URL of an image")
    if url:
        img_path = save_url_image(url)
        if img_path:
            # Display the downloaded image
            display_image = Image.open(img_path)
            st.image(display_image)
            # Feature extraction
            features = feature_extraction(img_path, model)
            # Recommendation
            indices = recommend(features, feature_list)
            # Show recommendations
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.subheader(i+1)
                    st.image(filenames[indices[0][i + 1]])
