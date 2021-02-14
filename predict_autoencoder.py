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

def create_data(data_path):
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

def get_data(X,Y):
  print("Training Data :\n")
  print("X Data :")
  x_data = create_data(BASE + X)
  display_details(x_data)
  
  print("Y Data :")
  y_data = create_data(BASE + Y)
  display_details(y_data)
  
  (trainX, testX, trainY, testY) = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
  return trainX, testX, trainY, testY

def get_predictions(autoencoder, filename, X_data, Y_data = []):
  print("\n\nResults :")
  n = min(15, len(X_data)-1)
  if not os.path.isdir("Images"):
    os.mkdir("Images")
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
    

def main(Y):
  print(Y,"\n\n")

  trainX, testX, trainY, testY = get_data(X,Y)
  autoencoder = get_best_model(Y)

  get_predictions(autoencoder, Y+"_train", trainX, trainY)
  get_predictions(autoencoder, Y+"_test", testX, testY)

  print("Kannada Data :")
  kannada_data = create_data("Fonts/Kannada_Fonts/Akhand")
  display_details(kannada_data)
  get_predictions(autoencoder, Y+"_kannada", kannada_data)

DIMS, CHANELS, EPOCHS, BASE, X = get_configs()
if __name__=="__main__":
  Y = sys.argv[1]
  main(Y)
