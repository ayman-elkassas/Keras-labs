from statistics import mode
from sklearn import neural_network
import tensorflow as tf

tf_v1 = tf.compat.v1

tf_v1.disable_eager_execution()

# todo: 1- read data

mnist = tf_v1.keras.datasets.mnist

# todo: 2- split data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print('X Train Shape is : ' , x_train.shape)
# print('X Train  is : ' , x_train[5])
# print('---------------------------------------- ')
# print('X Test Shape is : ' , x_test.shape)
# print('X Test  is : ' , x_test[5])
# print('---------------------------------------- ')
# print('y Train Shape is : ' , y_train.shape)
# print('y Train is : ' , y_train[5])
# print('---------------------------------------- ')
# print('y Test Shape is : ' , y_test.shape)
# print('y Test  is : ' , y_test[5])

# todo: 3- create model object (NN) with keras

NeuralNetwork = tf_v1.keras.models.Sequential()

# # todo: 4- add layers to model

# # todo: 4.1- add input layer (flatten for make 1D)
# # prepare input layer as flatten way that when inject x_data to model, it will be flatten
# # and number of units will be flexible with input data size (like: 28*28=784)
NeuralNetwork.add(tf_v1.keras.layers.Flatten())

# # todo: 4.2- add hidden layer
# # add hidden layer with units = 128
NeuralNetwork.add(tf_v1.keras.layers.Dense(128, activation=tf_v1.nn.relu))
# # activation =  softmax  , elu , relu , tanh , sigmoid , linear

NeuralNetwork.add(tf_v1.keras.layers.Dropout(0.2))

# # todo: 4.3- add output layer (softmax for make probability)

NeuralNetwork.add(tf_v1.keras.layers.Dense(10, activation=tf.nn.softmax))

# # todo: 4.4- compile model (optimizer, loss function, metrics) hyper parameters
NeuralNetwork.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# loss = sparse_categorical_crossentropy , binary_crossentropy , msest , mse , mae , kld , categorical_crossentropy

# # todo: 5- train model
NeuralNetwork.fit(x_train, y_train, epochs=3)

# # todo: 6- predict model
y_predict = NeuralNetwork.predict(x_test)

print('y_predict is : ', y_predict[0])

# # todo: 7- evaluate model

val_loss, val_acc = NeuralNetwork.evaluate(x_test, y_test)

print('val_loss is : ', val_loss)
print('val_acc is : ', val_acc)

# todo: 8- save model (should save model with the best accuracy to estimate in next time without train again)

# NeuralNetwork.save('1.model')

# todo: 9- load model

# new_model = tf_v1.keras.models.load_model('1.model')

# todo: 10- evaluate model (for get loss and accuracy)

# val_loss, val_acc = new_model.evaluate(x_test, y_test)
# print('val_loss is : ', val_loss)
# print('val_acc is : ', val_acc)

# todo: 11- can build model as one line

# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(128, activation=	tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#                                    ])
