from tensorflow import keras

# Installa TensorFlow

import tensorflow as tf
from hello_word import get_a_b

# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = create_model()

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=1)

# Save the weights
model.save('mnist_model.h5')

# # Create a new model instance
# model = create_model()
#
# # Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')

