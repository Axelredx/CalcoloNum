"""1
Compressione immagini SVD
• La funzione scipy.linalg.svd permette di calcolare la decomposizione SVD di una matrice.
• La libreria skimage permette di caricare/salvare immagini.
• Se skimage non risulta disponibile, si può installare eseguendo nell’Anaconda prompt il seguente
comando: conda install scikit-image
Exercise 1.1. Utilizzando la libreria skimage, nello specifico il modulo data, caricare e visualizzare un’im-
magine A (diversa dal cameraman) in scala di grigio di dimensione m × n.
1. Calcolare la matrice
    Ap =
         p
         £ ui ∗ viT ∗ σi
        i=1
    dove p ≤ rango(A)
2. Visualizzare l’immagine Ap .
3. Calcolare l’ errore relativo:
    ∥A − Ap ∥2 /∥A∥2
4. Calcolare il fattore di compressione
    cp = 1 /p * min(m, n) − 1.
    
5. Calcolare e plottare l’errore relativo e il fattore di compressione al variare di p.
Exercise 1.2. Eseguire l’esercizio precedente caricando un’immagine da un file usando 
la function skimage.io.imread.
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from skimage import data 
from skimage.io import imread

#A = data.astronaut()
#print(type(A))
#print(A.shape)
#A = A[:,:,2]
#plt.imshow(A, cmap='gray')
#plt.imshow(A)
#plt.show()

"""1.1"""
A = imread('phantom.png')
plt.imshow(A, cmap='grey')
plt.show()

"""1"""
U, s, Vh = sc.linalg.svd(A)

print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of V:', Vh.T.shape)

"""2"""
A_p = np.zeros(A.shape)
p_max = 10

for i in range(p_max):
  ui = U[:, i]
  vi = Vh[i, :]
  A_p = A_p + s[i]*np.outer(ui,vi)  ## =(u1v1 ... u1vn)
                                    ##  |  .   .   .  |
                                    ##  (unv1 ... unvn)

plt.imshow(A_p, cmap='grey')
plt.show()

"""3"""
err_rel = np.linalg.norm(A - A_p, ord=2) / np.linalg.norm(A, ord=2) #err assoluto sena la divisione

"""4"""
c = (1/p_max) * min(A.shape) -1

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è c=', c)

"""5"""
# al variare di p
p_max = 100
A_p = np.zeros(A.shape)
err_rel = np.zeros((p_max))
c = np.zeros((p_max))


for i in range(p_max):
  ui = U[:, i]
  vi = Vh[i, :]

  A_p = A_p + s[i]*np.outer(ui,vi)
  err_rel[i] = np.linalg.norm(A - A_p, ord=2) / np.linalg.norm(A, ord=2)
  c[i] = (1/(i+1)) * min(A.shape) -1 

plt.figure(figsize=(10, 5))

fig1 = plt.subplot(1, 2, 1)
fig1.plot(err_rel, 'o-')
plt.title('Errore relativo')

fig2 = plt.subplot(1, 2, 2)
fig2.plot(c, 'o-')
plt.title('Fattore di compressione')

plt.show()