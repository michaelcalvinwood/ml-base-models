import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# Custom Callbacks
class EarlyStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.85:
            print("\n\n85% validation accuracy has been reached.")
            self.model.stop_training = True

#####callback = EarlyStop()

# Data Loading

def retrieve_image_data(images_dir, batch_size=64, target_image_size=(224, 224), augmentation=None):
  # ImageDataGenerator is deprecated: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
  # Use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
  train_dir = images_dir + 'train/'
  test_dir = images_dir + 'test/'
  val_dir = images_dir + 'validation/'

  if augmentation is None:
    train_datagen = ImageDataGenerator(rescale=1./255)
  else:
    print('Using data augmentation')
    train_datagen = ImageDataGenerator(rescale=1./255, **augmentation)
    
  val_datagen = ImageDataGenerator(rescale=1./255)
  test_datagen = ImageDataGenerator(rescale=1./255)

  if not os.path.exists(train_dir):
    print("Error: training directory does not exist")
    return;

  input_shape = target_image_size + (3,)
    
  train_labels = os.listdir(train_dir)
  print(f"Labels: {train_labels}")
  num_labels = len(train_labels)
  if num_labels < 2:
    print("Error: Insufficient number of labels")
    return

  if num_labels==2:
    class_mode = "binary"
  else:
    class_mode = "categorical"

  print("\ntrain_data:")
  train_data = train_datagen.flow_from_directory(train_dir, batch_size=batch_size, target_size=target_image_size, class_mode=class_mode)

  if os.path.exists(val_dir):  
    val_labels = os.listdir(val_dir)
    print("\nval_data:")
    val_data = val_datagen.flow_from_directory(val_dir, batch_size=batch_size, target_size=target_image_size, class_mode=class_mode)
  else:
    print("No val_data")
    val_dir = None
    val_labels = None
    val_data = None

  if os.path.exists(test_dir):
    test_labels = os.listdir(test_dir)
    print("\ntest_data:")
    test_data = test_datagen.flow_from_directory(val_dir, batch_size=batch_size, target_size=target_image_size, class_mode=class_mode)
  else:
    print("No test_data")
    test_dir = None
    test_labels = None
    test_data = None
  
  return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_dir, val_dir, test_dir, input_shape, num_labels

# Data Analysis

def show_images_directory_stats(images_dir):
  for dirpath, dirnames, filenames in os.walk(images_dir):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def view_random_images(target_dir, num_images):
  """
  REWRITE THIS FUNCTION
  View num_images random images from the subdirectories of target_dir as a subplot.
  """
  # Get list of subdirectories
  subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]

  # Select num_images random subdirectories
  random.shuffle(subdirs)
  selected_subdirs = subdirs[:num_images]

  # Create a subplot
  fig, axes = plt.subplots(1, num_images, figsize=(15,9))
  for i, subdir in enumerate(selected_subdirs):
      # Get list of images in subdirectory
      image_paths = [f for f in os.listdir(os.path.join(target_dir, subdir))]
      # Select a random image
      image_path = random.choice(image_paths)
      # Load image
      image = plt.imread(os.path.join(target_dir, subdir, image_path))
      # Display image in subplot
      axes[i].imshow(image)
      axes[i].axis("off")
      axes[i].set_title(subdir)
  print(f"Shape of image: {image.shape}")    #width, height, colour channels
  plt.show()

# Data Preparation

def scalar_labels_to_one_hot_encoded(labels, num_classes='auto'):
  if num_classes == 'auto':
    num_classes = np.max(labels) + 1
  return tf.keras.utils.to_categorical(labels, num_classes)

# show images
def show_image(image, title=None):
  plt.figure()
  plt.imshow(image)
  plt.colorbar()
  plt.grid(False)
  if (title != None):
    plt.title(title)
  plt.show()

def show_indexed_images(images, indexes, labels):
  plt.figure(figsize=(9,9))
  for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel(f"{labels[indexes[i]]} ({indexes[i]})")
  plt.show()

# Plot Loss and Accuracy
def plot_loss_accuracy(history_1):
  # Extract the loss and accuracy history for both training and validation data
  loss = history_1.history['loss']
  val_loss = history_1.history['val_loss']
  acc = history_1.history['accuracy']
  val_acc = history_1.history['val_accuracy']

  # Create subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))

  # Plot the loss history
  ax1.plot(loss, label='Training loss')
  ax1.plot(val_loss, label='Validation loss')
  ax1.set_title('Loss history')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.legend()

  # Plot the accuracy history
  ax2.plot(acc, label='Training accuracy')
  ax2.plot(val_acc, label='Validation accuracy')
  ax2.set_title('Accuracy history')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.legend()

  plt.show()

def plot_loss_curves(history):
  keys = history.history.keys()

  if 'loss' in keys and 'val_loss' in keys and 'accuracy' in keys and 'val_accuracy' in keys:
    plot_loss_accuracy(history)
    return 

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

def show_confusion_matrix_for_binary_classification(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_pred = [0 if val < 0.5 else 1 for val in y_pred]
  eval_model=model.evaluate(X_test, y_test)
  print('Model Evaluation:\n', eval_model)
  print('\n\nConfusion Matrix:\n')
  eval = confusion_matrix(y_test, y_pred)
  print(eval)

def prepare_images_for_ANN(array):
  print(array.ndim)
  max = array[0].max()
  if max > 1:
    array = array / 255.0
  if array.ndim == 3:
    array = array.reshape(array.shape[0], array.shape[1]*array.shape[2])
  elif array.ndim == 4:
    array = array.reshape(array.shape[0], array.shape[1]*array.shape[2]*array.shape[3])
  return array, array[0].shape

def prepare_images_for_CNN(array):
  print(array.ndim)
  max = array[0].max()
  if max > 1:
    array = array / 255.0
  if array.ndim == 3:
    array = tf.expand_dims(array,axis=-1)
  return array, array[0].shape

def prepare_np_for_ANN(array):
  print(array.ndim)
  if array.ndim == 3:
    array = array.reshape(array.shape[0], array.shape[1]*array.shape[2])
  elif array.ndim == 4:
    array = array.reshape(array.shape[0], array.shape[1]*array.shape[2]*array.shape[3])
  return array, array[0].shape

def prepare_df_for_ANN(df, X_columns, y_column, test_size=0.2, random_state=42, dir="model_dump"):
  # Various Scalers: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
  df=df.dropna()

  # assign and save the X columns 
  X = df[X_columns]
  X_col_list = list(X.columns)
  # TODO: Save X_col_list list as JSON array so it can be used to ensure prediction columns are in the same order

  # Use one-hot encoding for all categorical X values
  for column in X_columns:
    if df[column].dtype == 'object':
      encoder = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")
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

def build_ANN(input_shape, layers=[10], flatten=False):
  model = Sequential()
  if flatten == True:
    model.add(Flatten(input_shape=input_shape))
  for index, units in enumerate(layers):
    print(units, index)
    if index == 0 and Flatten == False:
      model.add(Dense(units=units, input_shape=input_shape))
    else:
      model.add(Dense(units=units))
  return model

def build_CNN(input_shape, layers=[('c', 64),('p', 2)], flatten=True, dense=[], complex_head=[]):
  model = keras.models.Sequential()
  for index, layer in enumerate(layers):
    layer_type = layer[0]
    if layer_type == 'c':
      kernel_size = layer[1]
      if index == 0: 
        model.add(keras.layers.Conv2D(kernel_size, (3, 3), activation='relu', input_shape=input_shape))
      else:
        model.add(keras.layers.Conv2D(kernel_size, (3, 3), activation='relu'))
    elif layer_type == 'p':
      pool_size = layer[1]
      model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))
  model.add(tf.keras.layers.Flatten())

  if len(complex_head) > 0:
    for layers in complex_head:
      layer = layers[0]
      if layer == 'l2':
        model.add(tf.keras.layers.Dense(layers[1], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(layers[2])))
      elif layer == 'l1':
        model.add(tf.keras.layers.Dense(layers[1], activation="relu", kernel_regularizer=tf.keras.regularizers.l1(layers[2])))
      elif layer == 'fc':
        model.add(tf.keras.layers.Dense(layers[1], activation="relu"))
      elif layer == 'do':
        model.add(tf.keras.layers.Dropout(layers[1]))
                  

  if len(dense) > 0:
    for units in dense:
      model.add(tf.keras.layers.Dense(units, activation='relu'))
  return model

def run_model(model, X_train=None, y_train=None, outputs=1, epochs=100, X_validate=None, y_validate=None, verbose=0, loss='default', 
              optimizer='default', plot_loss=True, plot_accuracy=True, early_stop=True, monitor='loss', patience=5, min_delta=0,
              learning_rate=0.001, train_data=None, train_labels=None, validation_data=None):
  #optimizers = ['adam', 'rmsprop', 'sgd']
  # assign loss
  if loss == 'default':
    if outputs == 1:
      loss = 'mean_squared_error'
    elif outputs == 2:
      loss = 'binary_crossentropy'
    else:
      if y_train is not None:
        if y_train[0].ndim == 0:
          loss = 'sparse_categorical_crossentropy'
        else:
          loss = 'categorical_crossentropy'
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  # calculate units in output layer
  if outputs < 3:
    units = 1
  else:
    units = outputs

  num_layers = len(model.layers)
  
  # add output layer
  
  print('settings: ', f"\tactivation: {activation}\n", f"\toptimizer: {optimizer}\n", f"\tloss: {loss}\n", f"metrics: {metrics}")

  if num_layers > 0:
    model.add(Dense(units=units, activation=activation))
  else:
    model.add(Dense(units=units, activation=activation, input_shape=X_train[0].shape))
  
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  # add callbacks
  callbacks = []
  if early_stop:
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True))

  # fit
  if (X_validate is not None) and (y_validate is not None):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), verbose=verbose, callbacks=callbacks)
  elif X_validate is not None:
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose, callbacks=callbacks)
  elif (train_data is not None) and (validation_data is not None):
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs, verbose=verbose, callbacks=callbacks)
  else:
    history = model.fit(train_data, epochs=epochs, verbose=verbose, callbacks=callbacks)

  # Display the number of epochs used
  print(f"\n\nNumber of epochs trained for: {len(history.history['loss'])}")
  if metrics == 'accuracy' and X_validate is not None and y_validate is not None: 
    test_loss, test_acc=model.evaluate(X_validate, y_validate)
    print('Test Accuracy: ', test_acc)

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
import pathlib
import requests
def download_to_drive(path, filename, url):
  pathlib.Path(path).mkdir(parents=True, exist_ok=True)
  file_name = f"{path}{filename}"
  r = requests.get(url, stream = True)
  
  with open(file_name, "wb") as file:
      for block in r.iter_content(chunk_size = 1024): 
          if block:  
              file.write(block)

####download_to_google_drive('/content/drive/MyDrive/datasets/weather/', 'weather.zip', 'https://ml-datasets.nyc3.digitaloceanspaces.com/images_multi-classification/weather_01.zip')

import zipfile
def unzip_file(path, filename):
  file = f"{path}{filename}"
  with zipfile.ZipFile(file, 'r') as zip_ref:
      zip_ref.extractall(path)

##### Example: unzip_file('/content/drive/MyDrive/datasets/weather/', 'weather.zip')

# Model Evaluation

def evaluate_models(models, model_names,test_data):
    # REWRITE THIS FUNCTION
    import pandas as pd

    # Initialize lists for the results
    losses = []
    accuracies = []

    # Iterate over the models
    for model in models:
        # Evaluate the model
        loss, accuracy = model.evaluate(test_data)
        losses.append(loss)
        accuracies.append(accuracy)
       # Convert the results to percentages
    losses = [round(loss * 100, 2) for loss in losses]
    accuracies = [round(accuracy * 100, 2) for accuracy in accuracies]

    # Create a dataframe with the results
    results = pd.DataFrame({"Model": model_names,
                            "Loss": losses,
                            "Accuracy": accuracies})
    
    print(results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True).head())

    return results