import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist # Load data sets from keras

# Split data into training and testing
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # Tensor flow website

train_images = train_images / 255 # To get simpler values
test_images = test_images / 255

#plt.imshow(train_images[5], cmap=plt.cm.binary)
#plt.show()

# Creating a model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Flatten data
    keras.layers.Dense(128, activation="relu"), # Hidden layer
    keras.layers.Dense(10) # Output layer
])

# Compile the model
model.compile(optimizer="adam",
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
             metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
# test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
# print("Tested accuracy: ", test_accuracy)

# Making Predictions
prediction = model.predict(test_images)
#print(np.argmax(prediction[0]))

# Verfiying the model
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()