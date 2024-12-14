import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
import pandas as pd
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
from io import BytesIO

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))
csv_data = pd.read_csv('data.csv')

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
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

ids = []

# File upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                filename = os.path.basename(filenames[indices[0][i + 1]])
                st.subheader(f"{i+1}.")
                st.image(filenames[indices[0][i + 1]])
                st.text(f"Name: {filename}")
                file_id = csv_data.loc[csv_data['filename'] == filename, 'id'].values
                ids.append(file_id[0])
                if len(file_id) > 0:
                    st.text(f"ID: {file_id[0]}")
                else:
                    st.text("ID: Not found")

# URL input
else:
    url = st.text_input("Or enter the URL of an image")
    if url:
        img_path = save_url_image(url)
        if img_path:
            display_image = Image.open(img_path)
            st.image(display_image)
            features = feature_extraction(img_path, model)
            indices = recommend(features, feature_list)
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    filename = os.path.basename(filenames[indices[0][i + 1]])
                    st.subheader(f"{i+1}.")
                    st.image(filenames[indices[0][i + 1]])
                    st.text(f"Name: {filename}")
                    file_id = csv_data.loc[csv_data['filename'] == filename, 'id'].values
                    ids.append(file_id[0])
                    if len(file_id) > 0:
                        st.text(f"ID: {file_id[0]}")
                    else:
                        st.text("ID: Not found")

if ids:
    print(ids)
    url = "https://cozywear.free.beeceptor.com"
    requests.post(url, json={"ids": ids})
    