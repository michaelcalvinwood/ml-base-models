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
from sklearn.metrics import confusion_matrix

# Data Preparation

# Plot Loss and Accuracy
def plot_loss_curves(history):
  keys = history.history.keys()

  if 'loss' in keys: 
    loss = history.history['loss']
  else:
    return
  
  if 'val_loss' in keys: val_loss = history.history['val_loss']

  if 'accuracy' in keys: accuracy = history.history['accuracy']
  if 'val_accuracy' in keys: val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  if 'val_loss' in keys: plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  if 'accuracy' in keys: 
    plt.plot(epochs, accuracy, label='training_accuracy')
  else:
    return
  
  if 'val_accuracy' in keys: plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

# show confusion matrix
def show_confusion_matrix_for_binary_classification(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_pred = [0 if val < 0.5 else 1 for val in y_pred]
  eval_model=model.evaluate(X_test, y_test)
  print('Model Evaluation:\n', eval_model)
  print('\n\nConfusion Matrix:\n')
  eval = confusion_matrix(y_test, y_pred)
  print(eval)


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

  # numericize the y labels if needed
  
  if df[y_column[0]].dtype == 'object':
    uniques = df[y_column[0]].unique()
    num_uniques = len(uniques)
    if num_uniques == 2:
      label_encoder = LabelEncoder()
      labels = label_encoder.fit_transform(df[y_column[0]]) # numpy array
      y = pd.DataFrame({y_column[0]: labels})
      #y = labels

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

def run_model(model, X_train, y_train, outputs=1, epochs=100, X_validate=None, y_validate=None, verbose=0, loss='default', optimizer='default', plot_loss=True, plot_accuracy=True, early_stop=True):
  # assign loss
  if loss == 'default':
    if outputs == 1:
      loss = 'mean_squared_error'
    elif outputs == 2:
      loss = 'binary_crossentropy'
    else:
      loss = 'categorical_crossentropy'
  
  # assign output layer activation and metrics
  if outputs == 1:
    activation = None
    metrics = 'mae'
  elif outputs == 2:
    activation = 'sigmoid'
    metrics = 'accuracy'
  else:
    activation = 'softmax'
    metrics = 'accuracy'

  # assign optimizer
  if optimizer == 'default':
    optimizer = 'adam'

  num_layers = len(model.layers)
  
  # add output layer
  
  print('settings: ', activation, loss, optimizer, metrics)

  if num_layers > 0:
    model.add(Dense(units=1, activation=activation))
  else:
    model.add(Dense(units=1, activation=activation, input_shape=X_train[0].shape))
  
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  # add callbacks
  callbacks = []
  if early_stop:
    callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=4, min_delta=.0001, restore_best_weights=True))

  # fit
  if (X_validate is not None) and (y_validate is not None):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), verbose=verbose, callbacks=callbacks)
  else:
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose, callbacks=callbacks)
  
  # Plot Loss
  if plot_loss:
    plot_loss_curves(history)

  if outputs == 2 and X_validate is not None and y_validate is not None: show_confusion_matrix_for_binary_classification(model, X_validate, y_validate)
  
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