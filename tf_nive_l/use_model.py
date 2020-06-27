from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a basic model instance
mnist_model = tf.keras.models.load_model('mnist_model.h5')

folder = "numbers"
for filename in os.listdir(folder):
        path_img = os.path.join(folder,filename)
        if "jpg" in path_img:
            # predicting images
            img = image.load_img(path_img, target_size=(28, 28), color_mode = "grayscale")

            x = image.img_to_array(img)
            x = x.reshape((28, 28))
            x = np.expand_dims(x, axis=0)

            classes = mnist_model.predict(x)

            plt.imshow(img)
            plt.title("{}".format(classes[0]))
            plt.show()



print(classes)
