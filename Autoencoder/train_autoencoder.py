from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from imutils import paths
from keras import layers
from cv2 import cv2
import numpy as np
import keras
import sys
import os

def get_configs(attribute):
  configurations = {
                    "DIMS" : 256,
                    "CHANELS" : 1,
                    "EPOCHS" : 1000,
                    "BASE" : "Fonts/English_Fonts/",
                    "X" : "Screen_Sans_Normal",
                    "layer_depths" : [4, 8, 16, 32, 32, 16, 8, 4],
                    "layer_pools" : [2, 2, 2, 2, 2, 2, 2, 2],
                    "layer_convs" : [3, 3, 3, 3, 3, 3, 3, 3]
                    }
  return configurations[attribute]

def create_data(data_path):
  DIMS = get_configs("DIMS")
  CHANELS = get_configs("CHANELS")
  image_paths = sorted(list(paths.list_images(data_path)))
  print(image_paths)
  data = []
  for imagePath in image_paths:
    image = cv2.imread(imagePath)

    im = Image.open(imagePath)
    color=(255, 255, 255)
    im.load()  # needed for split()
    image = Image.new('RGB', im.size, color)
    image.paste(im, mask=im.split()[3])
    if CHANELS == 1:
      image = ImageOps.grayscale(image) 
    image = image.resize((DIMS,DIMS)) 

    image = img_to_array(image)
    data.append(image)
  data = np.array(data, dtype="float") / 255.0
  return data

def display_details(data):
  print(data.shape)
  # print(type(data[0]))
  # print(len(data))
  # plt.imshow(data[0])
  print()

def get_data(Y):
  BASE = get_configs("BASE")
  X = get_configs("X")
  print("Training Data :\n")
  print("X Data :")
  x_data = create_data(BASE + X)
  display_details(x_data)
  
  print("Y Data :")
  y_data = create_data(BASE + Y)
  display_details(y_data)
  
  (trainX, testX, trainY, testY) = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
  return trainX, testX, trainY, testY

def get_model():
  DIMS = get_configs("DIMS")
  CHANELS = get_configs("CHANELS")
  layer_depths = get_configs("layer_depths")
  layer_pools = get_configs("layer_pools")
  layer_convs = get_configs("layer_convs")
  print("\n\nModel :\n")
  input_img = keras.Input(shape=(DIMS, DIMS, CHANELS))

  size = len(layer_depths)

  x = layers.Conv2D(layer_depths[0], (layer_convs[0], layer_convs[0]), activation='relu', padding='same')(input_img)
  print("Conv2D        :", "{0:^4} :".format(layer_depths[0]), "{0:^4} :".format(layer_convs[0]), x.shape)
  for layer_index in range(1,size):
    # print(layer_index)
    x = layers.MaxPooling2D((layer_pools[layer_index-1], layer_pools[layer_index-1]), padding='same')(x)
    print("MaxPooling2D  :", "{0:^4} :".format(layer_pools[layer_index-1]), "     :", x.shape)
    x = layers.Conv2D(layer_depths[layer_index], (layer_convs[layer_index], layer_convs[layer_index]), activation='relu', padding='same')(x)
    print("Conv2D        :", "{0:^4} :".format(layer_depths[layer_index]), "{0:^4} :".format(layer_convs[layer_index]), x.shape)
  encoded = layers.MaxPooling2D((layer_pools[-1], layer_pools[-1]), padding='same')(x)
  print("MaxPooling2D  :", "{0:^4} :".format(layer_pools[-1]), "     :", encoded.shape)
  print("Encoder\n")

  x = layers.Conv2D(layer_depths[-1], (layer_convs[-1], layer_convs[-1]), activation='relu', padding='same')(encoded)
  print("Conv2D        : {0:^4} :".format(layer_depths[-1]), "{0:^4} :".format(layer_convs[-1]), x.shape)
  x = layers.UpSampling2D((layer_pools[-1], layer_pools[-1]))(x)
  print("UpSampling2D  :", "{0:^4} :".format(layer_pools[-1]), "     :", x.shape)
  for layer_index in range(size-2,-1,-1):
    # print(layer_index)
    x = layers.Conv2D(layer_depths[layer_index], (layer_convs[layer_index], layer_convs[layer_index]), activation='relu', padding='same')(x)
    print("Conv2D        :", "{0:^4} :".format(layer_depths[layer_index]), "{0:^4} :".format(layer_convs[layer_index]), x.shape)
    x = layers.UpSampling2D((layer_pools[layer_index], layer_pools[layer_index]))(x)
    print("UpSampling2D  :", "{0:^4} :".format(layer_pools[layer_index]), "     :", x.shape)
  decoded = layers.Conv2D(CHANELS, (3, 3), activation='sigmoid', padding='same')(x)
  print("Conv2D        :", "{0:^4} :".format(CHANELS), "{0:^4} :".format(3), decoded.shape)
  print("Decoder")

  autoencoder = keras.Model(input_img, decoded)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  return autoencoder

def train_model(autoencoder, trainX, testX, trainY, testY, Y):
  EPOCHS = get_configs("EPOCHS")
  print("\n\nTraining :")
  checkpointer = ModelCheckpoint(filepath=os.path.join("Models", Y + ".hdf5"),#'model_{epoch:03d}_{loss:.3f}_{val_loss:.3f}.hdf5'),
                                verbose=1,
                                save_best_only=True
                                )
  early_stopper = EarlyStopping(monitor='val_loss', mode='min', patience=30)
  history = autoencoder.fit(trainX,
                            trainY,
                            epochs=EPOCHS,
                            batch_size=32,
                            shuffle=True,
                            validation_data=(testX, testY),
                            callbacks=[early_stopper, checkpointer]
                            )
  return history

def plot_history(history, filename):
  print("\n\nHistory :")
  loss_train = history.history['loss']
  loss_val = history.history['val_loss']
  epochs = range(1,len(history.history['loss'])+1)
  plt.plot(epochs, loss_train, 'g', label='Training loss')
  plt.plot(epochs, loss_val, 'b', label='validation loss')
  plt.title('Training and Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig("Images/"+filename+".png")
  plt.show()

def get_predictions(autoencoder, filename, X_data, Y_data = []):
  DIMS = get_configs("DIMS")
  CHANELS = get_configs("CHANELS")
  print("\n\nResults :")
  n = min(15, len(X_data)-1)
  if len(Y_data) == 0:
    rows = 2
  else:
    rows = 3
  if CHANELS == 1:
    new_shape = (DIMS, DIMS)
  else:
    new_shape = (DIMS, DIMS, CHANELS)

  decoded_imgs = autoencoder.predict(X_data)
  plt.figure(figsize=(200, 40))
  for i in range(1, n + 1):
      # Display original
      ax = plt.subplot(rows, n, i)
      plt.imshow(X_data[i].reshape(new_shape))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # Display reconstruction
      ax = plt.subplot(rows, n, i + n)
      plt.imshow(decoded_imgs[i].reshape(new_shape))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      if rows == 3:
        # Display goal
        ax = plt.subplot(rows, n, i + n + n)
        plt.imshow(Y_data[i].reshape(new_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
  plt.savefig("Images/"+filename+".png")
  plt.show()

def get_best_model(Y):
    autoencoder = load_model("Models/" + Y + ".hdf5")
    return autoencoder
    
def main(Y, train):
  print(Y,"\n\n")
  if not os.path.isdir("Images"):
    os.mkdir("Images")
  trainX, testX, trainY, testY = get_data(Y)

  if train:
    autoencoder = get_model()
    history = train_model(autoencoder, trainX, testX, trainY, testY, Y)
    plot_history(history, Y+"_history")
  else:
    autoencoder = get_best_model(Y)

  get_predictions(autoencoder, Y+"_train", trainX, trainY)
  get_predictions(autoencoder, Y+"_test", testX, testY)

  print("Kannada Data :")
  kannada_data = create_data("Fonts/Kannada_Fonts/Akhand")
  display_details(kannada_data)
  get_predictions(autoencoder, Y+"_kannada", kannada_data)

if __name__=="__main__":
  Y = sys.argv[1]
  train = True
  if len(sys.argv) == 3:
    if sys.argv[2] == "test":
      train = False
  main(Y, train)
