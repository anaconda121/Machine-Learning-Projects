import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as numpy
from sklearn import linear_model, preprocessing
import pickle

"""
K-Nearest Neighbors - Classificiation Algorithm
	Algoritham that takes in an integer k and a test value. The algorithm relies upon 2 classes, which are
	generated from analysis of the dataset. The algorithm takes in the test value and places it in the 
	middle of the graph. Then, based on the number k, the algorithm will draw a circle around the test
	value until k amount of class reps are included in the circle. Next, the algorithm looks at the 
	reps and tallies the number from each class. Whichever class has a majority is the class that is
	classified to the test value.

	In other situations, the algorithm will split the graph into different parts, where each part is 
	associated to a certain class value

	Good Practices:
		k must always be an odd to prevent ties
		Always use features that only have numerical values - may need to convert non-nunmerical data into numerical data
"""

data = pd.read_csv('Datasets/car.data')

#use sklearn preprocessing to convert string data into ints
labels = preprocessing.LabelEncoder() #label encoder takes in lists, need to convert non-numerical data into lists, but rn we have pandas dataframe
buying = labels.fit_transform(list(data["buying"])) #takes all vals of buying col and transforms into a list with ints that correspond
maintainence = labels.fit_transform(list(data["maint"]))
doors = labels.fit_transform(list(data["door"]))
people = labels.fit_transform(list(data["persons"]))
lug_boot = labels.fit_transform(list(data["lug_boot"]))
safety = labels.fit_transform(list(data["safety"]))
Class = labels.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maintainence, doors, people, lug_boot, safety)) #x is features, attributes, zip converts attributes into tuples
y = list(Class) #y is what is trying to get predicted

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) 

best = 0
for _ in range(100):
	#implementing
	model = KNeighborsClassifier(n_neighbors = 7) #parameter is amount of neighbors
	model.fit(x_train, y_train)
	acc = model.score(x_test, y_test)

	if(acc > best):
		best = acc
		with open("car.pickle", "wb") as f:
			print("accuracy ",acc * 100) 
			pickle.dump(model, f)

pickle_in = open("car.pickle", "rb")
model = pickle.load(pickle_in)

prediction = model.predict(x_test)
#seeing classifier results
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(prediction)):
	distance = model.kneighbors([x_test[x]], 7,  True)
	print("prediction:", prediction[x], "Data: ", x_test[x], "Actual: ", y_test[x], "neighbor distance:", distance)
	print("prediction with string vals:", names[prediction[x]], "Data: ", x_test[x], "Actual: ", y_test[x], "\n", "neighbor distance:", distance)