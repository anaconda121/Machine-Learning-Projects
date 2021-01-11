% Initialize a matrix 
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
C = ones(2,3) %creates a 2 by three matrix with all values set to 1
C = 2 * ones(2,3) %creates a 2 by three matrix with all values set to 2
w = rand(2,3) %works for vectors as well, generates a 2 by 3 matrix with random values

% Initialize a vector 
v = [1;2;3]
v = 1:0.1:2 %intializes a vector that has a first value of one and future elements incement by 1 until last value = 2 
v = 1:6 %v is now a vector that starts from 1 and increments by until last value = 6

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)

% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

rows = size(A,1) %gets number of rows in matrix A
cols = size(A,2) %gets number of cols in matrix A

% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]