import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, InceptionV3, Xception, ResNet50
from PIL import Image


# image_1 = 'https://static.wikia.nocookie.net/silicon-valley/images/4/49/Jian_Yang.jpg'
# image_2 = 'http://www.semantics3.com/blog/content/images/downloaded_images/hot-dog-and-a-not-hot-dog-the-distinction-matters-code-included-8550067fb16/1-VrpXE1hE4rO1roK0laOd7g.png'
# image_3 = './assets/silicon_valley.jpg'
# image_4 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRd0abE3OlR-F7Su6Hrv8q_Wf-3J4ZGzY2KzQ&usqp=CAU'
# url_1 = 'https://www.youtube.com/watch?v=ACmydtFDTGs'
hnh = './assets/hnh.png'

st.title("Hotdog, Not-Hotdog")
col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)

st.header('')  # for proper spacing with slider, etc.
pick_model = col2.radio('Pick a Learning-Transfer Model', ("Xception","MobileNetV2", "InceptionV3", "ResNet50", "VGG16"), index=0)
sensitivity = col2.slider('Sensitivity (0 = all hotdogs, 100 = no hotdogs)', 0, 100, 50)
uploaded_file = col2.file_uploader("Upload an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner(text="  \nClassifying..."):
        # initialize Transfer-Learning Model
        train_gen_aug = ImageDataGenerator(
            rescale = 1/255.,
            shear_range=0.1,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip=True
        )

        ### MODEL SET HERE ###
        if(pick_model == 'Xception'):
            conv_base = Xception(include_top=False, input_shape=(299, 299, 3))
            reloaded_model = tf.keras.models.load_model('./models/hotdog_model_1.h5')
        elif(pick_model == 'MobileNetV2'):
            conv_base = MobileNetV2(include_top=False, input_shape=(299, 299, 3))
            reloaded_model = tf.keras.models.load_model('./models/hotdog_model_2.h5')
        elif(pick_model == 'InceptionV3'):
            conv_base = InceptionV3(include_top=False, input_shape=(299, 299, 3))
            reloaded_model = tf.keras.models.load_model('./models/hotdog_model_3.h5')
        elif(pick_model == 'ResNet50'):
            conv_base = ResNet50(include_top=False, input_shape=(299, 299, 3))
            reloaded_model = tf.keras.models.load_model('./models/hotdog_model_4.h5')
        elif(pick_model == 'VGG16'):
            conv_base = VGG16(include_top=False, input_shape=(299, 299, 3))
            reloaded_model = tf.keras.models.load_model('./models/hotdog_model_5.h5')

        img_arr = np.array(image)
        img_arr = img_arr.reshape(1, 299, 299, 3)
        img_aug = train_gen_aug.flow(img_arr)
        p = conv_base.predict(img_aug)
        preds = reloaded_model.predict(p)

        offset = 0
        if(sensitivity == 0):
            offset = 1
        if(sensitivity == 100):
            offset = -1
        adj_sens = 1 - ((sensitivity + offset) / 100)

        if(preds[0][0] <= adj_sens):
            hnh = './assets/hotdog.png'
        else:
            hnh = './assets/nothotdog.png'

        import base64
        from io import BytesIO
        st.markdown(
            """
            <style>
            .container {
                display: flex;
                position: relative;
                top: -50px;
            }
            .hnh-img {
                position: relative;
                top: 100px;
                z-index:2;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue())

        col1.markdown(
            f"""
            <div class="container">
                <div>
                <img class="hnh-img" src="data:image/png;base64,{base64.b64encode(open(hnh, "rb").read()).decode()}">
                <div>
                </div>
                <img src="data:image/png;base64,{image_base64.decode()}">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


