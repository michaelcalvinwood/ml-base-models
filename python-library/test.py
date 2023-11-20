from ez_keras import ez_keras as ez
import numpy as np
import tensorflow as tf

X = np.arange(-100, 100, 4)
y = X + 10

X = tf.expand_dims(X, axis=-1)

model = ez()
model.build_ANN(X[0].shape, [10])
model.train(X, y, outputs=1)