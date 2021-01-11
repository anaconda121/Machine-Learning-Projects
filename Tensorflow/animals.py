import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import matplotlib.pyplot as plt 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os 
import numpy as np 

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''
# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

#creating network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #first parm is num filters, second is size of feature box
model.add(layers.MaxPooling2D((2, 2))) #param is size of feature box, used to condense data
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #we don't need to specify shape because network will find shape from previous layers
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) 
history = model.fit(train_images, train_labels, epochs = 4, validation_data = (test_images, test_labels))  #validation data has model also get tested on test data, and model is better there b/c it has already seen imgs before
test_loss, test_acc = model.evaluate(test_images,  test_labels)
print(test_acc)'''

'''
rn, accuracy is low b/c dataset is very small. Solution is data-augmentation, which is the theory that an image can be flipped,
rotated, and manipulated and all of these images can be passed into the network. This way, our data has grown exponentially, and 
the model will become better because it will still be able to find patterns among the images. 
'''

#different modes of data augmentation
datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)

# pick an image to transform
test_img = train_images[20]
img = image.img_to_array(test_img)  # convert image to numpy arry
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 5 images
        break

plt.show()

'''
another solution is a pretrained model. Companies such as google have created opensource networks that have been trained on 
millions of images, so we can use those models are the base of our own model and finetune some details.
'''

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str # creates a function object that we can use to get labels

# display 5 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#resize images

img_size = 160
def format_example(image, label):
	image = tf.cast(image, tf.float32) #convert all pixel vals into float
	image = (image / 127.5) - 1
	image = tf.image.resize(image, (img_size, img_size))
	return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

'''
for image, label in train.take(2):
	plt.figure()
	plt.imshow(image, cmap = plt.cm.binary)
	plt.title(get_label_name(label))
'''

#randomize data and group it together for inputting into network
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#picking a pretrained model
IMG_SHAPE = (img_size, img_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet') #include top = do we want to include the classifier that comes with this model. we don't because we are using the model specifically for cats and dogs, not the 1000 classes that it has been trained for

#freezing base b/c we don't want to retrain model and change the weights and biases b/c they have already been defined and they work well
base_model_trainable = False

#adding classifer

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
output_layer = keras.layers.Dense(1) # creating one dense nuetron b/c we only have to predict for 2 classes
model = tf.keras.Sequential([
	base_model,
	global_average_layer,
	output_layer
])	

#training model
base_learning_rate = 0.0001  #learning rate is scale of how much we can change weights and biases,small here b/c base model has already been trained for us
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

epochs = 1
validation_steps=20

history = model.fit(train_batches, epochs=epochs, validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

#saving model
model.save("catsanddogs.h5")
new_model = tf.keras.models.load_model('catsanddogs.h5')