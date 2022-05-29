import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf_v1 = tf.compat.v1
tf_v1.disable_eager_execution()

mnist = tf_v1.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

print(y_train[0]) 

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

model.save('kerasNN.model')
new_model = tf.keras.models.load_model('kerasNN.model')
predictions = new_model.predict(x_test)
print(predictions)

print(np.argmax(predictions[2]))
plt.imshow(x_test[2],cmap=plt.cm.binary)
plt.show()