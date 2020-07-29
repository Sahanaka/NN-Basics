# Classification of movie reviews using the imdb data set

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the data set
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()

word_index = {k:(v + 3) for k, v in word_index.items()} # The dicionary
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Make data similar in shape
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250) # Making reviews to a length of 250 characters
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250) # Making reviews to a length of 250 characters

# Decoding function
def decoding(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

'''
# Defining the model
model = keras.Sequential()

# Adding layers
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# # Model summary
# print(model.summary())

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Splitting training data into validation data
x_validation = train_data[:10000] # 10000 for validation
x_train = train_data[10000:]

y_validation = train_labels[:10000] 
y_train = train_labels[10000:]

# Train the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_validation, y_validation), verbose=1)

loss, accuracy = model.evaluate(test_data, test_labels)
print(loss, accuracy)

# Predicting
test_review = test_data[0]
prediction = model.predict([test_review])

# print("Review")
# print(decoding(test_review))
# print("Prediction: " + str(prediction[0]))
# print("Actual: " +  str(test_labels[0]))

# Saving the model
model.save("model.h5")
'''

# Load the saved model - Comment the rest
model = keras.models.load_model("model.h5")

def review_encode(string):
    encoded = [1]

    for word in string:
        if word.lower() in word_index:
             encoded.append(word_index[word.lower()])
        else:
             encoded.append(2)
        
    return encoded


# Making predictions to outside data
with open('test.txt', encoding="utf-8") as file:
    for line in file.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)        
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict =  model.predict(encode)
        print(line)
        print(encode)
        print(predict[0]) 

