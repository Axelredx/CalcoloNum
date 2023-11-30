"""Exercise 1.4. Ripetere i punti precedenti utilizzando anche l’operatore downsampling con i seguenti
fattori di scaling sf = 2,4,8,16."""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft
#from utils import psf_fft, A, AT, gaussian_kernel   # commentato per pt.2
from utils_SR import psf_fft, A, AT, gaussian_kernel
# Immagine in floating point con valori tra 0 e 1
X = data.camera().astype(np.float64)/255 #conversione in float + divisione x ottenere val tra 0 e 1 dei vari pixel (normalizzazione dati)
m, n = X.shape
sf = 4

# Genera il filtro di blur
k = gaussian_kernel(24,3)
plt.imshow(k)
plt.show()

# Blur with FFT
K = psf_fft(k,24,X.shape)
plt.imshow(np.abs(K))
plt.show()


X_blurred = A(X,K,sf) #sf è fattore di scailing

# Genera il rumore
sigma = 0.02
np.random.seed(42)
noise = np.random.normal(size=X_blurred.shape) * sigma

# Aggiungi blur e rumore
y = X_blurred + noise
ATy = AT(y, K, sf)
PSNR = metrics.peak_signal_noise_ratio(X,ATy) #uso ATy x avere stesse dimensioni di immagini di partenza
mse = metrics.mean_squared_error(X,ATy) 


plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
#plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(X_blurred, cmap='gray', vmin=0, vmax=1)
plt.title('Downsampled')
#plt.xticks([]), plt.yticks([])
plt.show()

# Visualizziamo i risultati
plt.figure(figsize=(30, 10))
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
#plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title(f'Corrupted (PSNR: {PSNR:.2f})')
#plt.xticks([]), plt.yticks([])
plt.show()

#########################Soluzione naive
from scipy.optimize import minimize

# Funzione da minimizzare
def f(x):
    x = x.reshape((m, n))
    Ax = A(x, K,sf)
    return 0.5 * np.sum(np.square(Ax - y))

# Gradiente della funzione da minimizzare
def df(x):
    x = x.reshape((m, n))
    ATAx = AT(A(x,K,sf),K, sf)
    d = ATAx - ATy
    return d.reshape(m * n)

# Minimizzazione della funzione
x0 = ATy.reshape(m*n)
max_iter = 25
res = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_iter, 'return_all':True})

# Per ogni iterazione calcola il PSNR rispetto all'originale
PSNR = np.zeros(max_iter + 1)
for k, x_k in enumerate(res.allvecs):
    PSNR[k] = metrics.peak_signal_noise_ratio(X, x_k.reshape(X.shape))

# Risultato della minimizzazione
X_res = res.x.reshape((m, n))

# PSNR dell'immagine corrotta rispetto all'oginale
starting_PSNR = np.full(PSNR.shape[0], metrics.peak_signal_noise_ratio(X, ATy))

# Visualizziamo i risultati
ax2 = plt.subplot(1, 2, 1)
ax2.plot(PSNR, label="Soluzione naive")
ax2.plot(starting_PSNR, label="Immagine corrotta")
plt.legend()
plt.title('PSNR per iterazione')
plt.ylabel("PSNR")
plt.xlabel('itr')
plt.subplot(1, 2,2).imshow(X_res, cmap='gray', vmin=0, vmax=1)
plt.title('Immagine Ricostruita')
plt.xticks([]), plt.yticks([])
plt.show()

# Regolarizzazione
# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m, n))
    Ax = A(x, K, sf)
    return 0.5 * np.sum(np.square(Ax - y)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m, n)
    ATAx = AT(A(x,K, sf),K, sf)
    d = ATAx - ATy
    return d.reshape(m * n) + Lx

x0 = ATy.reshape(m*n)
lambdas = [0.01,0.03,0.04, 0.06]
PSNRs = []
images = []

# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X.shape)
    images.append(X_curr)

    # Stampa il PSNR per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X, X_curr)
    PSNRs.append(PSNR)
    print(f'PSNR: {PSNR:.2f} (\u03BB = {L:.2f})')
    
    

# Visualizziamo i risultati
plt.plot(lambdas,PSNRs)
plt.title('PSNR per $\lambda$')
plt.ylabel("PSNR")
plt.xlabel('$\lambda$')
plt.show()

plt.figure(figsize=(30, 10))

plt.subplot(1, len(lambdas) + 2, 1).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title("Originale")
plt.xticks([]), plt.yticks([])
plt.subplot(1, len(lambdas) + 2, 2).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title("Corrotta")
plt.xticks([]), plt.yticks([])


for i, L in enumerate(lambdas):
  plt.subplot(1, len(lambdas) + 2, i + 3).imshow(images[i], cmap='gray', vmin=0, vmax=1)
  plt.title(f"Ric. ($\lambda$ = {L:.2f})")
plt.show()

"""Exercise 1.5 (Facoltativo TV). Un’altra funzione adatta come termine di regolarizzazione `e la Variazione
Totale. Data x immagine di dimensioni m ×n la variazione totale TV di x `e definita come:
TV (x) =
n∑
i
m∑
j
√
||∇x(i,j)||22 + ε2 (4)
Come nei casi precedenti il problema di minimo che si va a risolvere `e il seguente:
x∗ = arg minx
1
2 ||Ax −b||22 + λTV (x) (5)
il cui gradiente ∇f `e dato da
∇f(x) = (ATAx −ATb) + λ∇TV (x) (6)
Utilizzando il metodo del gradiente e la funzione minimize, calcolare la soluzione del precendente
problema di minimo regolarizzato con la funzione TV per differenti valori di λ, utilizzando le funzioni
totvar e grad totvar.
Per calcolare il gradiente dell’immagine ∇u usiamo la funzione ‘np.gradient‘ che approssima la derivata
per ogni pixel calcolando la differenza tra pixel adiacenti. I risultati sono due immagini della stessa
dimensione dell’immagine in input, una che rappresenta il valore della derivata orizzontale e l’altra della
derivata verticale . Il gradiente dell’immagine nel punto (i,j) `e quindi un vettore di due componenti,
uno orizzontale contenuto e uno verticale.
Per risolvere il problema di minimo `e necessario anche calcolare il gradiente della variazione totale che
`e definito nel modo seguente
∇TV (u) = −div
(
∇u√||∇u||22 + ε2
)
(7)
dove la divergenza `e definita come
div(F) = ∂Fx
∂x + ∂Fy
∂y (8)
div(F) `e la divergenza del campo vettoriale F, nel nostro caso F ha due componenti dati dal gradiente
dell’immagine ∇u scalato per il valore 1√||∇u||22+ε2. Per calcolare la divergenza bisogna calcolare la
derivata orizzontale ∂Fx
∂x della componente x di F e sommarla alla derivata verticale ∂Fy
∂y della com-
ponente y di F. Per specificare in quale direzione calcolare la derivata con la funzione ‘np.gradient‘
utilizziamo il parametro ‘axis = 0‘ per l’orizzontale e ‘axis = 1‘ per la verticale """

#####################opzionale!!!!!! (NON finito)

# Regolarizzazione
# Funzione da minimizzare
def f(x, L):
    #nsq = np.sum(np.square(x))
    x  = x.reshape((m, n))
    nsq = totvar(x) #variazione totale
    Ax = A(x, K, sf)
    return 0.5 * np.sum(np.square(Ax - y)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m, n)
    ATAx = AT(A(x,K, sf),K, sf)
    d = ATAx - ATy
    return d.reshape(m * n) + Lx

x0 = ATy.reshape(m*n)
lambdas = [0.01,0.03,0.04, 0.06]
PSNRs = []
images = []

# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X.shape)
    images.append(X_curr)

    # Stampa il PSNR per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X, X_curr)
    PSNRs.append(PSNR)
    print(f'PSNR: {PSNR:.2f} (\u03BB = {L:.2f})')
    
    

# Visualizziamo i risultati
plt.plot(lambdas,PSNRs)
plt.title('PSNR per $\lambda$')
plt.ylabel("PSNR")
plt.xlabel('$\lambda$')
plt.show()

plt.figure(figsize=(30, 10))

plt.subplot(1, len(lambdas) + 2, 1).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title("Originale")
plt.xticks([]), plt.yticks([])
plt.subplot(1, len(lambdas) + 2, 2).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title("Corrotta")
plt.xticks([]), plt.yticks([])


for i, L in enumerate(lambdas):
  plt.subplot(1, len(lambdas) + 2, i + 3).imshow(images[i], cmap='gray', vmin=0, vmax=1)
  plt.title(f"Ric. ($\lambda$ = {L:.2f})")
plt.show()
