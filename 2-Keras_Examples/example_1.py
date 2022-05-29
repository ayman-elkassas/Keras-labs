import tensorflow as tf

tf_v1 = tf.compat.v1
tf_v1.disable_eager_execution()


mnist= tf.compat.v1.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #scale manually (may be use scale from sklearn)

model= tf.compat.v1.keras.models.Sequential([
    tf.compat.v1.keras.layers.Flatten(input_shape=(28,28)),
    tf.compat.v1.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.compat.v1.keras.layers.Dropout(0.2), #remove 20% of neurons
    tf.compat.v1.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# # todo: 6- predict model
y_predict = model.predict(x_test)

print('y_predict is : ', y_predict[0])

val_loss, val_acc =model.evaluate(x_test, y_test)

print('val_loss is : ', val_loss)
print('val_acc is : ', val_acc)
