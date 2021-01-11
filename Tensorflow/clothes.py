import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist #using keras to load dataset 

(train_images, train_labels), (test_images, test_labels) = data.load_data() #split into train and test data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#right now, the images come in arrays of 28 by 28, and all together, this is a lot of data, we need to condense images onto a scale of 0 and 1, instead of 0 - 255
train_images = train_images / 255.0 #works b/c images are stored in a numpy array
test_images = test_images / 255.0

#however, nueron will not be able to accurately process a 28 by 28 array, so we need to flatten the array and make it 1-D, now array will have 784 indexes 
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(128 , activation = "relu"), #dense means connect all nuerons from previous layer, activation is for activation layer
	keras.layers.Dense(10, activation = "softmax") #softmax is probability of answer being right 
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_images, train_labels, epochs = 5) #epochs mean how many times will the model see the info passed in 0

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("accuracy", test_acc)

prediction = model.predict(test_images)

for i in range (5):
	plt.grid(False)
	plt.imshow(train_images[i], cmap = plt.cm.binary)
	plt.xlabel("actual: " + class_names[test_labels[i]])
	plt.title("prediction: "+ class_names[np.argmax(prediction[i])]) #predictions are percentages that the model thinks correspond to the classes
	plt.show()