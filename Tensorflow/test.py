import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)


word_index = data.get_word_index() 

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

rating_length = np.arange(len(train_data))
for i in range(len(train_data)):
	rating_length[i] = len(train_data[i])

largest_rating = 0
for i in range(len(rating_length)):
	if rating_length[i] > largest_rating:
		largest_rating = rating_length[i]

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=largest_rating)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=largest_rating)


model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")  # name it whatever you want but end with .h5

test_review = test_data[0]
predict = model.predict([test_review])
print("review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)