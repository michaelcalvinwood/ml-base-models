from ez_keras import ez_keras as ez
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore') 

X = np.arange(-100, 100, 4)
y = X + 10

X_train, input_shape = ez.prepare_np_for_ANN(X)
y_train = y

model = ez()
model.build_ANN(input_shape, [100, 100, 50])
model.train(X_train, y_train, outputs=1)

#model.show_keras()