import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

'''
Support Vector Machines - classification algo

Generates a line called a hyperplane that is a line that acts as a boundary for the 2 classes.
For every d-dimensions, the hyperplane is d-1 dimensions. The distance between the 2 closest
points from each class is maximized when drawing a hyperplane. 

When data of 2 classes is scattered: We can use a kernel (function) to transform the graph 
into another dimension. This can help with drawing the hyperplane. Kernel Function examples:
f(x) = x* y, etc. Kernel returns dot product of vectors (multiplying 2 vectors). Kernel also helps 
with reducing computational space usage. 

Pro: effective in high dimensional spaces, easy to use
Con: will return bad results in there is not enough data

Soft margin: hyperplane line where the parts may have vals of both classes, can improve classification
hard magin: opposite of a soft margin
'''


cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target #0 represents malingnant, 1 represents benign

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.15) #splits data into 4 arrays, 10% of data is saved for testing
classes = ['malingnant', 'benign']

clf = KNeighborsClassifier(n_neighbors  = 9)
#clf = svm.SVC(kernel = "linear", C = 2) #c is margin, 0 = hard margin, 1 = soft margin
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print( acc)

