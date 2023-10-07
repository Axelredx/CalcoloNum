import numpy as np
"""HOW TO NUMPY

Numpy is a multi dimensional array library (list slow = bad :( ) and uses contigues memory

-Ex of code:

a = np.array([[1,2,3],[4,5,6]], dtype='int32') (tipo di dato omettibile)

np.ndim(a) -> 2  dim array
shape(,,) Mrighe, Ncolonne, Zprofondità
np.shape(a) -> (2,3)

a[r,c] -> elem row r, col c
a[r,:] -> all elem row r
a[:,c] -> all elem col c
a[r, startindex:endindex:stepsize]

a[r,c]=x -> substitute elem with x
a[:,c]=x -> sub all in col c with x

#3D ARRAY
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

a[x,y,z] -> x = num tuplas, y = num row, z = num col
es: a[0,1,1] -> 4
a[:,1,:] -> [[3,4],[7,8]]

#INITIALIZE ALL TYPES OF ARRAY

np.zeros(5) -> array of five 0
np.ones((2,3)) -> matrix 2x3 of 1
np.full((2,2),99) -> matrix 2x2 full of 99

np.random.randint(startindex-1,stopindex+1, size=(r,c)) -> rxc with random num between si-1,sti+1
np.identity(5) -> identity matrix 5x5

#COPY ARRAY
b = a.copy() -> otherwise if i change an elem of b i'll be changed also in a cause of pointers

#MATHEMATICS/LINEAR ALGEBRA
+,-,*,/,** to all elem of a

np.matmul(a,b) -> matrix multi (col a = row b!!)
np.linalg.det(a) -> determinante
np.linalg.norm(a,x) -> norma, x= 1,2,'fro'(frobius),np.inf
np.linalg.cond(a,x) -> condizionam, x= 1,2,'fro'(frobius),np.inf
    """
  
a = np.array([[1,2,3],[4,5,6]])  
print(np.shape(a))

b = np.ones((5,5))
print(b)
c = np.zeros((3,3))
print(c)
c[1,1] = 9
print(c)
b[1:4,1:4] = c
print(b)

#[[1. 1. 1. 1. 1.]
 #[1. 0. 0. 0. 1.]
 #[1. 0. 9. 0. 1.]
 #[1. 0. 0. 0. 1.]
 #[1. 1. 1. 1. 1.]]
