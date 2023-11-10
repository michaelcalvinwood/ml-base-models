import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler

# Data Preparation

# Various Scalers: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
def prepare_df_for_ANN(df, X_columns, y_column, test_size=0.2, random_state=42, dir="model_dump"):
  df=df.dropna()

  # assign and save the X columns 
  X = df[X_columns]
  X_col_list = list(X.columns)
  # TODO: Save X_col_list list as JSON array so it can be used to ensure prediction columns are in the same order

  # Use one-hot encoding for all categorical X values
  for column in X_columns:
    if df[column].dtype == 'object':
      encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
      one_hot_encoded=encoder.fit_transform(X[[column]])
      X = pd.concat([X,one_hot_encoded],axis=1).drop(columns=[column])
      # TODO: Save the encoder here

  # assign the y column
  y = df[y_column]

  # split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  # scale the X data
  new_X_col_list = list(X_train.columns)
  for column in new_X_col_list:
      max = X_train[column].max()
      min = X_train[column].min()
      if min < 0 or max > 1:
        encoder = MinMaxScaler().set_output(transform="pandas")
        X_train[column] = encoder.fit_transform(X_train[[column]])
        X_test[column] = encoder.transform(X_test[[column]])
        # TODO: Save Scaler

  # TODO: If y column is categorical then one hot encode y column

  input_shape = X_train.values[0].shape

  return X_train, X_test, y_train, y_test, input_shape

# Building and Running Models

def build_ANN(input_shape, layers=[10]):
  model = Sequential()
  for index, units in enumerate(layers):
    print(units, index)
    if index == 0:
      model.add(Dense(units=units, input_shape=input_shape))
    else:
      model.add(Dense(units=units))
  return model

def run_model(model, X_train, y_train, outputs=1, epochs=100, X_validate=None, y_validate=None, verbose=0, loss='default', optimizer='default', plot_loss=True, plot_accuracy=True):
  # assign loss
  if loss == 'default':
    if outputs == 1:
      loss = 'mean_squared_error'
    else:
      loss = 'categorical_crossentropy'

  # assign optimizer
  if optimizer == 'default':
    optimizer = 'adam'

  num_layers = len(model.layers)
  
  # add output layer
  if outputs == 1:
    if num_layers > 0:
      model.add(Dense(units=1))
    else:
      model.add(Dense(units=1, input_shape=X_train[0].shape))

  # compile
  model.compile(optimizer=optimizer, loss=loss)

  # fit
  if (X_validate is not None) and (y_validate is not None):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), verbose=verbose)
  else:
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose)
  
  # Plot Loss
  if plot_loss:
    if (X_validate is not None) and (y_validate is not None):
      print(history.history.keys())
    else:
      #print(history.history.keys())
      # Lets plot the loss
      plt.plot(history.history['loss'])
      plt.title('Model loss')
      plt.ylabel('Loss')
      plt.xlabel('Number of epochs')
      plt.legend(['loss plot'], loc='upper right')
      plt.show()
  
  return model, history

def get_value_prediction(model, input):
  return model.predict(input)[0][0]
  



# Visualizing Data
def line_plot(X, y, xlabel='', ylabel='', title=''):
  plt.plot(X, y)
  if xlabel: plt.xlabel(xlabel)
  if ylabel: plt.ylabel(ylabel)
  if title: plt.title(title)
  plt.show()

#line_plot(X, y, 'Hours of Study', 'Test Score', 'Test Scores from Hours of Study')