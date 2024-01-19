import numpy as np

"""1. Write a NumPy program to create a 3x3 matrix with values ranging from 2 to
10. Go to the editor
Expected Output:
[[ 2 3 4]
 [ 5 6 7]
 [ 8 9 10]]
    """
a = np.full((3,3),2)
a[0,:] = np.arange(2,5)
a[1,:] = np.arange(5,8)
a[2,:] = np.arange(8,11)
print(a)

"""2. Write a NumPy program to create a 2d array with 1 on the border and 0
inside
    """
a = np.ones((3,3))
a[1,1]=0
print(a)

"""3. Write a NumPy program to find common values between two arrays.
    """
a = np.arange(10)
b = np.linspace(3,12,10)
print(a,b)
for i in a:
    for j in b:
        if(i == j):
            cond = True
if(cond):
    print(">:)")
    
"""4. Write a NumPy program to get the values and indices of the elements that
are bigger than 10 in a given array. Go to the editor
Original array:
[[ 0 10 20]
 [20 30 40]]"""
a = np.array([[0,10,20],[20,30,40]])
for i in range(2):
    for j in range(3):
        if(a[i,j]>=10):
            print("i:",i," j:",j," num:",a[i,j])
        
"""5. Write a NumPy program to create a new shape to an array without changing
its data. Go to the editor
Reshape 3x2:
[[1 2]
 [3 4]
 [5 6]]
Reshape 2x3:
[[1 2 3]
 [4 5 6]]"""
a = np.array([[1,2],[3,4],[5,6]])
print(a)
b = a.reshape(2,3)
print(b)