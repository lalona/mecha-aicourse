# Installa TensorFlow

import tensorflow as tf

def get_a_b():
    a = 5
    b = 6
    return a, b

if __name__ == "__main__":
    a, b = get_a_b()


    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = tf.keras.models.Sequential([
      # 784
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=32)

    model.evaluate(x_test,  y_test, verbose=1)