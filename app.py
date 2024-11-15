import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

# Define the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Load existing data if it exists
if os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl'):
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
else:
    feature_list = []
    filenames = []

# Convert filenames to a set for quick lookup
filenames_set = set(filenames)

# Detect changes in the images folder
current_files = set(os.path.join('images', f) for f in os.listdir('images'))

# Add new files
new_files = current_files - filenames_set
for file in tqdm(new_files, desc="Adding new files"):
    filenames.append(file)
    feature_list.append(extract_features(file, model))

# Remove deleted files
deleted_files = filenames_set - current_files
for file in tqdm(deleted_files, desc="Removing deleted files"):
    idx = filenames.index(file)
    filenames.pop(idx)
    feature_list.pop(idx)

# Save updated data
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)
