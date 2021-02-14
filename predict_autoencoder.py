import sys
from functions import *

def main(Y):
  print(Y,"\n\n")

  trainX, testX, trainY, testY = get_data(Y)
  autoencoder = get_best_model(Y)

  get_predictions(autoencoder, Y+"_train", trainX, trainY)
  get_predictions(autoencoder, Y+"_test", testX, testY)

  print("Kannada Data :")
  kannada_data = create_data("Fonts/Kannada_Fonts/Akhand")
  display_details(kannada_data)
  get_predictions(autoencoder, Y+"_kannada", kannada_data)

if __name__=="__main__":
  Y = sys.argv[1]
  main(Y)
