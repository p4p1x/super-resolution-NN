import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, add, Dropout
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

dataset_dir = 'input/Image Super Resolution - Unsplash'
highres_dir = os.path.join(dataset_dir, 'high res')
lowres_dir = os.path.join(dataset_dir, 'low res')

data = pd.read_csv(os.path.join(dataset_dir, "image_data.csv"))
data['low_res'] = data['low_res'].apply(lambda x: os.path.join(lowres_dir, x))
data['high_res'] = data['high_res'].apply(lambda x: os.path.join(highres_dir, x))

train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)

batch_size = 2
target_size = (800, 1200)


def preprocess_image(img_path, target_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255
    return img


def preprocess_pair(low_res_path, high_res_path, target_size):
    return (preprocess_image(low_res_path, target_size),
            preprocess_image(high_res_path, target_size))


def create_dataset(subset_data, target_size, batch_size):
    low_res_paths = subset_data['low_res'].values
    high_res_paths = subset_data['high_res'].values

    ds = tf.data.Dataset.from_tensor_slices((low_res_paths, high_res_paths))
    ds = ds.map(lambda x, y: preprocess_pair(x, y, target_size))
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    return ds, len(subset_data) // batch_size


train_dataset, train_steps = create_dataset(train_data, target_size, batch_size)
val_dataset, val_steps = create_dataset(val_data, target_size, batch_size)

for low_res, high_res in train_dataset.take(3):
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].imshow(low_res[0])
    axs[0].set_title('Obraz w niskiej rozdzielczości')
    axs[1].imshow(high_res[0])
    axs[1].set_title('Obraz w wysokiej rozdzielczości')
    plt.show()

input_img = Input(shape=(800, 1200, 3))
x1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
x3 = MaxPooling2D(padding='same')(x2)
x3 = Dropout(0.3)(x3)
x4 = Conv2D(128, (3, 3), padding='same', activation='relu')(x3)
x5 = Conv2D(128, (3, 3), padding='same', activation='relu')(x4)
x6 = MaxPooling2D(padding='same')(x5)
x7 = Conv2D(256, (3, 3), padding='same', activation='relu')(x6)
x8 = UpSampling2D()(x7)
x9 = Conv2D(128, (3, 3), padding='same', activation='relu')(x8)
x10 = Conv2D(128, (3, 3), padding='same', activation='relu')(x9)
x11 = add([x5, x10])
x12 = UpSampling2D()(x11)
x13 = Conv2D(64, (3, 3), padding='same', activation='relu')(x12)
x14 = Conv2D(64, (3, 3), padding='same', activation='relu')(x13)
x15 = add([x14, x2])
decoded = Conv2D(3, (3, 3), padding='same', activation='relu')(x15)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.summary()

model_path = 'autoencoder_batch2_epoch9.h5'

callbacks_list = [ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                  EarlyStopping(monitor='val_loss', min_delta=0, patience=9, verbose=1, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.2, min_lr=0.00000001)]

hist = autoencoder.fit(train_dataset,
                       steps_per_epoch=train_steps,
                       validation_data=val_dataset,
                       validation_steps=val_steps,
                       epochs=9,
                       callbacks=callbacks_list)

plt.figure(figsize=(20, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

for low_res, high_res in val_dataset.take(5):
    pred = autoencoder.predict(low_res)
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    axs[0].imshow(low_res[0])
    axs[0].set_title('Obraz w niskiej rozdzielczości')
    axs[1].imshow(high_res[0])
    axs[1].set_title('Obraz w wysokiej rozdzielczości')
    axs[2].imshow(pred[0])
    axs[2].set_title('Przewidywany obraz w wysokiej rozdzielczości')
    plt.show()
