import numpy as np 

a = np.array([[1,2,3,4], [5,6,7,8]])
print(a)

#index slicing
b = a [0, :] #first param is rows, second is cols print out entire first row
print("\n", b)

b = a [-1, :] #-1 = last row
print("\n", b)

#index slicing with booleans
bool_indx = a > 2 #represents val w/ true if index > 2
print("\n", bool_indx) 
print("\n", a[bool_indx]) #prints all vals that are greater than 2 in array

b = np.where(a > 2, a, -1) #if a < 2, replaces val with -1
print("\n", b)

#reshaping arrays
a = np.arange(1,7)
print("\n", a)
print(a.shape)

b = a.reshape((2,3))
print("\n", b)
print(b.shape)

#np.newaxis can convert arrays into row / column vectors
b = a[np.newaxis, :] 
print("\n", b)
print(b.shape)

b = a[:, np.newaxis]
print("\n", b)
print(b.shape)