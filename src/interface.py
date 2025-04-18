# ייבוא ספריות
from iris_vector_normalization import IrisVectorNormalization
from tensorflow.keras import Model
from numpy.linalg import norm
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os


# מודל טריפלט
class TripletLoss:
    def __init__(self):
        self.inp = tf.keras.Input(shape=(None, None, 3))
        self.out = []

    def embedding(self):
        vgg16_fe = tf.keras.applications.VGG16(
            include_top=False, weights=None, input_tensor=self.inp
        )
        vgg16_fe.trainable = False
        out_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=1, strides=1, padding='same',
            activation='relu', name='out_conv')(vgg16_fe.layers[-1].output)
        self.out = tf.keras.layers.GlobalMaxPooling2D()(out_conv)
        return Model(inputs=vgg16_fe.inputs, outputs=self.out, name='embedding')

# הורדת משקולות מהאינטרנט
def download_weights(url):
    import tempfile
    response = requests.get(url)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

# פונקציית עיבוד תמונה
def preprocess(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    w_target = 400
    w_orig, h_orig = image.size
    h_target = int((w_target / w_orig) * h_orig)
    image_resized = image.resize((w_target, h_target))
    img = IrisVectorNormalization(image_resized)
    if img is None:
        return None
    img = np.array(img).astype("float32")
    return np.expand_dims(img, axis=0)

model_copy = TripletLoss()
embedding = model_copy.embedding()

# חובה: הרצת דוגמה לבניית המודל
embedding.build((None, 75, 400, 3))

weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "final.weights.h5")
embedding.load_weights(weights_path)

st.title("🔒 Iris Authentication System")

img_file1 = st.file_uploader("Upload Image 1 jpg file only*", type=["jpg"])
img_file2 = st.file_uploader("Upload Image 2 jpg file only*", type=["jpg"])

if img_file1 and img_file2:
    img1 = preprocess(img_file1)
    img2 = preprocess(img_file2)

    if img1 is None or img2 is None:
        st.error("❌ Could not process one or both images.")
    else:
        # תצוגת התמונות
        st.image([img1.squeeze() / 255.0, img2.squeeze() / 255.0],
                 caption=["Image 1", "Image 2"], width=150)

        # חישוב מרחק והחלטה
        embedding1 = embedding.predict(img1)
        embedding2 = embedding.predict(img2)
        distance = norm(embedding1 - embedding2)

        st.write("🔍 **Distance between embeddings:**", float(distance))
        threshold = 1.49
        if distance < threshold:
            st.success("✅ Authentication Successful")
        else:
            st.warning("❌ Authentication Failed")