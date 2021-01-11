import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle #saving models - if model has a long train time, pickle can help us b/c we don't have to train model every time

"""
Linear Regression - a straight line of best fit genereated by that values of 
x and y values. Positive relationship is both independant (x) and dependent
(y) values increase. Negative relationship if independent val increases
but dependent val decreases. 

Computer generates a line of best fit in multi-dimensional space, where dimensions
are specified by number of attributes

Formula: b0 + b1 * x == y= mx + b
	b0 = y-intercept
	b1 = slope
	x = x-value

When not to use Linear Regression
	Linear Regression is bad when data is scattered around the graph and there is no correlation.
	This is because the model does not know what direction the points are headed in, and as 
	a result, is unable to draw a accurate line
"""

data = pd.read_csv("Datasets/student-mat.csv", sep = ";") #sep  is delimeter

#trimming data down
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #gets parameters specified in list

predict = "G3"

X = np.array(data.drop([predict], 1)) #removing g3 from dataset b/c we want to predict it, x holds features
y = np.array(data[predict]) #y holds actual data for column

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) #splits data into 4 arrays, 10% of data is saved for testing

#saving best model using pickle
best = 0
for _ in range(100):
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) #splits data into 4 arrays, 10% of data is saved for testing

	#creating training model
	linear = linear_model.LinearRegression()
	linear.fit(x_train, y_train)
	accuracy = linear.score(x_test, y_test) #finding accuracy of model

	if(accuracy > best):
		best = accuracy
		#save model
		with open("studentmodel.pickle", "wb") as f:
			print("accuracy ",accuracy * 100) 
			pickle.dump(linear, f)

prediction = linear.predict(x_train)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficent: \n', linear.coef_) #generates 5 coefficents because we have 5 cols
print('Intercept: \n', linear.intercept_)

#implementing model
prediction = linear.predict(x_train)
'''
for i in range(len(prediction)):
	print(prediction[i], x_test[i], y_test[i]) #first val is prediction, 2nd val is input values from dataset, third value is actual final grade from dataset 
'''

#graphing model
p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data['G3'], color = 'gray')
plt.plot(x_train, prediction, color = 'red', linewidth = 2)
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()