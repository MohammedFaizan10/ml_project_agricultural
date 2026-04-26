import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pickle
from tensorflow.keras.models import model_from_json
from sklearn.tree import DecisionTreeRegressor
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage import color
import scipy.stats as stats
import os

# --- Constants & Mappings ---
labels_Soil = ['Alluvial soil', 'Black soil','Clay soil', 'Red soil']
crops = ['Rice,Wheat, Sugarcane,Maize,Cotton', 'Virginia, Wheat,Jowar,Millets','Rice,Broccoli,Cabbage', 'Cotton,Pulses,Potatoes,OilSeeds']

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

leaf_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 
               'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
               'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

fertilizers = ['Mancozeb or Captan ', 'Copper-based fungicide', 'Sulfur or Lime Sulfur ', 
               'No fertilizer needed ', 'Azoxystrobin ', 'Mancozeb for Common Rust', 
               'No fertilizer needed', 'Chlorothalonil ', 'Copper-based fungicide ', 
               'Thiophanate-methyl ', 'No fertilizer needed ', 'Mancozeb ', 
               'Copper oxychloride ', 'No fertilizer needed ', 
               'Metalaxyl or Mancozeb ', 'Copper-based fungicide ', 
               'Chlorothalonil ', 'No fertilizer needed ', 'Metalaxyl or Chlorothalonil ', 
               'Chlorothalonil ', 'Mancozeb for Septoria ', 
               'Insecticidal soap or Abamectin', 'Difenoconazole', 
               'Copper-based fungicide', 'Imidacloprid or Acetamiprid']

crop_mapping = {
    1.0: "Rice", 2.0: "Wheat", 3.0: "Mung Bean", 4.0: "Tea",
    5.0: "Millet", 6.0: "Maize", 7.0: "Lentil", 8.0: "Jute"
}

# --- Helper Functions ---
def channels_first_transform(image):
    return image.transpose((2,0,1))

def remove_green_pixels(image):
    channels_first = channels_first_transform(image)
    r_channel = channels_first[0]
    g_channel = channels_first[1]
    b_channel = channels_first[2]
    mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
    channels_first = np.multiply(channels_first, mask)
    image = channels_first.transpose(1, 2, 0)
    return image

def rgb2gray(image):
    return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False):
    single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
    gclm = graycomatrix(single_channel_image, offsets, angles)
    return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image):
    image = channels_first_transform(image).reshape(3,-1)
    r_hist = np.histogram(image[0], bins = 26, range=(0,255))[0]
    g_hist = np.histogram(image[1], bins = 26, range=(0,255))[0]
    b_hist = np.histogram(image[2], bins = 26, range=(0,255))[0]
    return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
    color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
    return np.array([
        np.mean(color_histogram), np.std(color_histogram),
        stats.entropy(color_histogram), stats.kurtosis(color_histogram),
        stats.skew(color_histogram), np.sqrt(np.mean(np.square(color_histogram)))
    ])

def glcm_features(glcm):
    return np.array([
        graycoprops(glcm, 'correlation'), graycoprops(glcm, 'contrast'),
        graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'),
        graycoprops(glcm, 'dissimilarity'),
    ]).flatten()

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
    image = remove_green_pixels(full_image) if remove_green else full_image
    gray_image = rgb2gray(image)
    glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
    return glcm_features(glcmatrix)

def extract_features(image):
    offsets=[1,3,10,20]
    angles=[0, np.pi/4, np.pi/2]
    channels_first = channels_first_transform(image)
    return np.concatenate((
        texture_features(image, offsets=offsets, angles=angles),
        texture_features(image, offsets=offsets, angles=angles, remove_green=False),
        histogram_features_bucket_count(image),
        histogram_features(channels_first[0]),
        histogram_features(channels_first[1]),
        histogram_features(channels_first[2]),
    ))


# --- Load Models ---
@st.cache_resource
def load_all_models():
    # Soil CNN Model
    with open('model/model1.json', "r") as json_file:
        cnn_classifier = model_from_json(json_file.read())
    cnn_classifier.load_weights("model/model_weights1.h5")
    
    # Leaf CNN Model
    with open('model/model.json', "r") as json_file:
        cnn_model = model_from_json(json_file.read())
    cnn_model.load_weights("model/model_weights.h5")
    
    # PCA Model
    with open('model/pca.txt', 'rb') as file:
        pca = pickle.load(file)
        
    # Train Decision Tree Model on the fly using cpdata
    df = pd.read_csv('cpdata.csv')
    X_train = df.iloc[:, 0:7].values
    y_train = df.iloc[:, 8].values
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    
    return cnn_classifier, cnn_model, pca, dt_model

# Streamlit App Execution
st.set_page_config(page_title="Agriculture ML Tool", page_icon="🌱", layout="wide")
st.title("🌱 Optimizing Agriculture using Machine Learning")

with st.spinner("Loading AI Models..."):
    soil_model, leaf_model, pca, crop_model = load_all_models()

tabs = st.tabs(["🌾 Crop Prediction", "🌿 Leaf Disease Detection", "🪨 Soil Classification"])

# ----------------------------
# 1. CROP PREDICTION TAB
# ----------------------------
with tabs[0]:
    st.header("Predict the Best Crop based on Soil Data")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature", value=25.0)
        humidity = st.number_input("Humidity", value=80.0)
        ph = st.number_input("pH Level", value=6.5)
        rainfall = st.number_input("Rainfall (mm)", value=200.0)
    with col2:
        n = st.number_input("Nitrogen (N)", value=0.5)
        p = st.number_input("Phosphorus (P)", value=0.5)
        k = st.number_input("Potassium (K)", value=0.5)
        
    if st.button("Predict Crop", type="primary"):
        input_data = np.array([temp, humidity, ph, rainfall, n, p, k]).reshape(1, -1)
        pred = crop_model.predict(input_data)[0]
        crop_name = crop_mapping.get(pred, "Unknown Crop")
        st.success(f"**Recommended Crop to Grow:** {crop_name}")


# ----------------------------
# 2. LEAF DISEASE DETECTION TAB
# ----------------------------
with tabs[1]:
    st.header("Detect Leaf Disease from Image")
    leaf_file = st.file_uploader("Upload Leaf Image", type=['jpg', 'png', 'jpeg'], key="leaf_img")
    
    if leaf_file is not None:
        file_bytes = np.asarray(bytearray(leaf_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)
        
        if st.button("Detect Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                img_resized = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
                imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(imgHSV, low_green, high_green)
                mask = 255 - mask
                res = cv2.bitwise_and(img_resized, img_resized, mask=mask)
                res_ravel = res.ravel()
                
                test = np.asarray([res_ravel])
                test_pca = pca.transform(test)
                test_pca = test_pca.astype('float32') / 255.0
                
                preds = leaf_model.predict(test_pca)
                predict_idx = np.argmax(preds)
                
                st.error(f"**Detected Disease:** {leaf_labels[predict_idx]}")
                st.success(f"**Recommended Fertilizer/Treatment:** {fertilizers[predict_idx]}")


# ----------------------------
# 3. SOIL CLASSIFICATION TAB
# ----------------------------
with tabs[2]:
    st.header("Classify Soil Type & Recommend Crop")
    soil_file = st.file_uploader("Upload Soil Image", type=['jpg', 'png', 'jpeg'], key="soil_img")
    
    if soil_file is not None:
        file_bytes = np.asarray(bytearray(soil_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)
        
        if st.button("Classify Soil", type="primary"):
            with st.spinner("Extracting features..."):
                img_resized = cv2.resize(img, (64,64))
                features = extract_features(img_resized)
                
                test = np.asarray([features])
                test = test.astype('float32') / 255.0
                test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
                
                preds = soil_model.predict(test)
                predict_idx = np.argmax(preds)
                
                st.info(f"**Predicted Soil Type:** {labels_Soil[predict_idx]}")
                st.success(f"**Crops Recommended for this soil:** {crops[predict_idx]}")