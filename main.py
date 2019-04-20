from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


import sys
import pandas as pd
import numpy as np
import tqdm
import os
from keras import Sequential
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

## custom imports
# from plots import *
from inception import getInceptionV3
from captions import * 


# Needed packages
# get_ipython().system('pip install tensorflow')
# get_ipython().system('pip install keras')
# get_ipython().system('pip install --upgrade numpy')

def getSpearmanCorScore(Y_pred,Y_true):
    # Calculate the Spearmann"s correlation coefficient
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
      print('Input shapes don\'t match!')
      return
    if len(Y_pred.shape) == 1:
      Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})
      score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)
      return score_mat.iloc[1][0]
    # else:
    #   print("spearman", Y_pred.shape[1])
    #   for ii in range(Y_pred.shape[1]):
    #     getSpearmanCorScore(Y_pred[:,ii],Y_true[:,ii])

def train_model(df, oneHotWords):
  Y = df[['short-term_memorability','long-term_memorability']].values
  X = oneHotWords
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)
  nb_iterations = 8

  # add dropout
  # add regularizers
  ## divided by 10 all regularizers values
  ## TODO: change activations params
  print("input_shape: ", len(oneHotWords[0]))
  model = Sequential()
  ## TODO: add inception feature, change input_shape ?
  model.add(layers.Dense(12,activation='relu',kernel_regularizer=regularizers.l2(0.0001), input_shape=(len(oneHotWords[0]),)))
  model.add(layers.Dropout(0.6))
  model.add(layers.Dense(12,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(2,activation='sigmoid'))

  # compile the model, changed from rmsprop to adagrad
  #Todo: try to change loss & metrics options (and some others ?)
  model.compile(optimizer='adagrad',loss='mse',metrics=['accuracy'])
  print("X train: ", X_train.shape)
  print("X test: ", X_test.shape)
  print("Y_train: ", Y_train.shape)
  print("Y_test: ", Y_test.shape)
  # training the model // epochs value 20 => 8
  history = model.fit(X_train,Y_train,epochs=nb_iterations,validation_data=(X_test,Y_test))

  # visualizing the model
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  ##len(loss) == nb_iteration ?
  epochs = range(1,len(loss)+1)

  # plt.plot(epochs,loss,'bo',label='Training loss')
  # plt.plot(epochs,val_loss,'b',label='Validation loss')
  # plt.title('Training and validation loss')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.legend()
  # plt.show()

  # plt.figure()
  # acc = history.history['acc']
  # val_acc = history.history['val_acc']
  # plt.plot(epochs, acc, 'bo', label='Training acc')
  # plt.plot(epochs, val_acc, 'b', label='Validation acc')
  # plt.title('Training and validation accuracy')
  # plt.xlabel('Epochs')
  # plt.ylabel('Acc')
  # plt.legend()
  # plt.show()


  predictions = model.predict(X_test)
  
  print('Short term Spearman\'s correlation coefficient is: %.3f' % getSpearmanCorScore(predictions[:,0],Y_test[:,0]))
  print('Long term Spearman\'s correlation coefficient is: %.3f' % getSpearmanCorScore(predictions[:,1],Y_test[:,1]))

def main():
  # vidId, short term score, short term annotations, long term score, long term annotation
  ground_truth_file = './dev-set/dev-set/dev-set_ground-truth.csv'
  ground_truth = pd.read_csv(ground_truth_file)
#  ground_truth.describe()
  # drawPlots(ground_truth)

  ## inception
  ## captions
  vidCapts = getVideosCaptions()
  df = getPredAndCaptions(ground_truth, vidCapts)
  getInceptionV3(df)
  oneHotWords = countWordsOccur(df)
  print("oneHotWords shape:", oneHotWords.shape)


  # The targets here are the short-term and long-term scores
  train_model(df, oneHotWords)



  # Good, now we can index by filename and get the array of 1000 elements quickly.
  # Note that in this this data frame we are now treating the video number as index

if __name__ == "__main__":
  main()