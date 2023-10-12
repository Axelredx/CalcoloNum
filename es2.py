"""Matrici e norme
Exercise 1.1. Si consideri la matrice A

    A =([[1, 2], [0.499, 1.001]])

• Calcolare la norma 1, la norma 2, la norma Frobenius e la norma infinito di A con 
numpy.linalg.norm() (guardare l’help della funzione)."""

import numpy as np
import sys

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])
print ('Norme di A:')
norm1 = np.linalg.norm(A,1)
norm2 = np.linalg.norm(A,2)
normfro = np.linalg.norm(A,'fro')
norminf = np.linalg.norm(A,np.inf)

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')

"""• Calcolare il numero di condizionamento di A con numpy.linalg.cond() 
(guardare l’help della funzione).
"""

cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

"""• Considerare il vettore colonna x = (1, 1)T
e calcolare il corrispondente termine noto b per il sistema
lineare Ax = b."""

#x = np.array([[1], [1]])
x = np.ones((2, 1))
#x = np.array([[1, 1]]).T
# Ax = b
b = A.dot(x)
print ('b=',b,'\n')

"""
• Considerare ora il vettore ˜b = (3, 1.4985)T
e verifica che ˜x = (2, 0.5)T `e soluzione del sistema Ax˜ = ˜b"""

btilde = np.array([[3], [1.4985]])
#xtilde = np.array([[2], [0.5]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
print ('A*xtilde = ', A.dot(xtilde) , '\n')

"""
• Calcolare la norma 2 della perturbazione sui termini noti ∆b = ∥b−˜b∥2 e la norma 2 della perturbazione
sulle soluzioni ∆x = ∥x − x˜∥2. Confrontare ∆b con ∆x."""

deltax = np.linalg.norm(x-xtilde, ord=2)
deltab = np.linalg.norm(b-btilde, ord=2)

print ('delta x = ', deltax)
print ('delta b = ', deltab)

##################################################
##################################################
##################################################
##################################################

"""Metodi diretti
Exercise 2.1. Si consideri la matrice
    A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
• Creare il problema test in cui il vettore della soluzione esatta `e x = (1, 1, 1, 1)T
e il vettore termine
noto `e b = Ax.
"""
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
import scipy.linalg.decomp_lu as LUdec 
# help (LUdec)
# help(scipy.linalg.lu_solve )

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones(A.shape[1])
b = np.matmul(A,x)

condA = np.linalg.cond(A,1)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

"""• Guardare l’help del modulo scipy.linalg.decomp lu e usare una delle sue funzioni per calcolare la
fattorizzazione LU di A con pivoting. Verificare la correttezza dell’output.
    """
#help(LUdec.lu_factor)
lu, piv =LUdec.lu_factor(A)

print('lu',lu,'\n')
print('piv',piv,'\n')

"""
• Risolvere il sistema lineare con la funzione lu solve del modulo decomp lu oppure con la funzione
scipy.linalg.solve triangular.
• Stampare la soluzione calcolata e valutarne la correttezza.  
NB L’inversa di una matrice viene calcolata con la funzione np.linalg.inv
    """
# risoluzione di    Ax = b   <--->  PLUx = b 
my_x=LUdec.lu_solve((lu,piv),b)

print('my_x = \n', my_x)
print('norm =', scipy.linalg.norm(x-my_x))

# verifica
print('\nSoluzione calcolata: ')
for i in range(n):
    print('%0.2f' %my_x[i])
    
#ERRORE RELATIVO
err=np.linalg.norm(my_x-x,2)/np.linalg.norm(x)
print('%e',err)

##################################################
##################################################
    
"""Exercise 2.2. 
Si ripeta l’esercizio precedente sulla matrice di Hilbert, che si pu`o generare con la funzione
A = scipy.linalg.hilbert(n) per n = 5, . . . , 10. In particolare:
• Calcolare il numero di condizionamento di A e rappresentarlo in un grafico al variare di n
    """
import matplotlib.pyplot as plt

# crazione dati e problema test
K_A=np.zeros((6,1))
for n in np.arange(5,11):
    A=scipy.linalg.hilbert(n)
    x=np.ones((A.shape[1],1))
    b=np.matmul(A,x)
    K_A[n-5]=np.linalg.cond(A)
    
    print('x: \n', x , '\n')
    print('x.shape: ', x.shape, '\n' )
    print('b: \n', b , '\n')
    print('b.shape: ', b.shape, '\n' )
    print('A: \n', A, '\n')
    print('A.shape: ', A.shape, '\n' )
    print('K(A)=', K_A[n-5], '\n')


x = np.arange(5,1)
plt.plot(x,K_A,color='blue', linestyle='--')
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K_A')
plt.show()

"""• Considerare il vettore colonna x = (1, . . . , 1)T , calcola il corrispondente termine noto 
b per il sistema lineare Ax = b e la relativa soluzione x̃ usando la fattorizzazione di Cholesky 
come nel caso precedente."""
Err=np.zeros((6,1))

for i in arange(5,11):
    # decomposizione di Choleski
    L = scipy.linalg.cholesky (i)
    print('L:', L, '\n')
    print('L.T*L =', np.matmul(L, np.transpose(L)), '\n')
    Err[i-5] = scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro')
    print('err = ', Err[i-5], '\n')    
    y = ...
    my_x = ...
    print('my_x = \n ', my_x)

    print('norm =', scipy.linalg.norm(x-my_x, 'fro'))