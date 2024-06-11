import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model('autoencoder_batch2_epoch9.h5')

def load_image(image_path, target_size=(1200, 800)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255
    return img_arr

def display_images(lowres_img, highres_img):
    fig, axs = plt.subplots(1,2,figsize=(20,5))
    axs[0].imshow(lowres_img)
    axs[0].set_title('Zdjęcie o niskiej rozdzielczości')
    axs[1].imshow(highres_img)
    axs[1].set_title('Przewidywane zdjęcie o wysokiej rozdzielczości')
    plt.show()

img_path = 'gory.jpg'
lowres_img = load_image(img_path)
highres_img = model.predict(lowres_img)

lowres_img_shape = (lowres_img[0]*255).astype(np.uint8)
highres_img_shape = (highres_img[0]*255).astype(np.uint8)
display_images(lowres_img_shape, highres_img_shape)
