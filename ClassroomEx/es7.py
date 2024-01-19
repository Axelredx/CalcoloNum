"""1 Image deblur
Il problema di deblur consiste nella ricostruzione di un immagine a partire da un dato acquisito mediante il
seguente modello:
    y = Ax + η (1)
    dove :
    y rappresenta l’immagine corrotta,
    x rappresenta l’immagine originale che vogliamo ricostruire
    A rappresenta l’operatore che applica il blur Gaussiano
    η ∼ N(0,σ2) rappresenta una realizzazione di rumore additivo con distribuzione Gaussiana di media
    μ = 0 e deviazione standard σ
Exercise 1.1. Problema test
        Caricare l’immagine camera dal modulo skimage.data rinormalizzandola nel range [0,1].
        Applicare un blur di tipo gaussiano con deviazione standard 3 il cui kernel ha dimensioni 24 ×24.
    utilizzando la funzione. Utilizzare prima cv2 (open-cv) e poi la trasformata di Fourier.
        Aggiungere rumore di tipo gaussiano, con σ = 0.02, usando la funzione np.random.normal().
        Calcolare le metriche Peak Signal Noise Ratio (PSNR) e Mean Squared Error (MSE) tra l’immagine
    degradata e l’immagine esatta usando le funzioni peak signal noise ratio e mean squared error
    disponibili nel modulo skimage.metrics.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft
from utils import psf_fft, A, AT, gaussian_kernel 

# Immagine in floating point con valori tra 0 e 1
X = data.camera().astype(np.float64)/255 #conversione in float + divisione x ottenere val tra 0 e 1 dei vari pixel (normalizzazione dati)
m, n = X.shape

# Genera il filtro di blur
k = gaussian_kernel(24,3)
plt.imshow(k)
plt.show()

# Blur with openCV
X_blurred = cv.filter2D(X,-1,k)
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(X_blurred, cmap='gray', vmin=0, vmax=1)
plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# Blur with FFT
K = psf_fft(k,24,X.shape)
plt.imshow(np.abs(K))
plt.show()

X_blurred = A(X,K)

# Genera il rumore
sigma = 0.02
np.random.seed(42)
noise = np.random.normal(size=X.shape) * sigma

# Aggiungi blur e rumore
y = X_blurred + noise
PSNR = metrics.peak_signal_noise_ratio(X,y) #maggiore è e più vicine sono le immagini
mse = metrics.mean_squared_error(X,y) #minore è e più vicine sono le immagini == PSNR
ATy = AT(y, K)


# Visualizziamo i risultati
plt.figure(figsize=(30, 10))
plt.subplot(121).imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122).imshow(y, cmap='gray', vmin=0, vmax=1)
plt.title(f'Corrupted (PSNR: {PSNR:.2f})')
plt.xticks([]), plt.yticks([])
plt.show()

"""Exercise 1.2. Soluzione naive Una possibile ricostruzione dell’immagine originale x partendo dall’imma-
gine corrotta y `e la soluzione naive data dal minimo del seguente problema di ottimizzazione:
        x∗= argminx 1/2 ∥Ax −y∥22(2)
    Utilizzando il metodo del gradiente coniugato implementato dalla funzione minimize della libreria
    scipy, calcolare la soluzione naive.
    Analizza l’andamento del PSNR e dell’MSE al variare del numero di iterazion
"""
# Soluzione naive
from scipy.optimize import minimize

# Funzione da minimizzare
def f(x):
    x = x.reshape((m, n))
    Ax = A(x, K)
    return 0.5 * np.sum(np.square(Ax - y))

# Gradiente della funzione da minimizzare
def df(x):
    x = x.reshape((m, n))
    ATAx = AT(A(x,K),K)
    d = ATAx - ATy
    return d.reshape(m * n)

# Minimizzazione della funzione
x0 = y.reshape(m*n)
max_iter = 25
res = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_iter, 'return_all':True})

# Per ogni iterazione calcola il PSNR rispetto all'originale
PSNR = np.zeros(max_iter + 1)
for k, x_k in enumerate(res.allvecs):
    PSNR[k] = metrics.peak_signal_noise_ratio(X, x_k.reshape(X.shape))

# Risultato della minimizzazione
X_res = res.x.reshape((m, n))

# PSNR dell'immagine corrotta rispetto all'oginale
starting_PSNR = np.full(PSNR.shape[0], metrics.peak_signal_noise_ratio(X, y))

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

"""Exercise 1.3. Soluzione regolarizzata Si consideri il seguente problema regolarizzato secondo Tikhonov
        x∗= argminx 1/2 ∥Ax −y∥22+ λ∥x∥22(3)
    Utilizzando sia il metodo del gradiente che il metodo del gradiente coniugato calcolare la soluzione del
        problema regolarizzato.
    Analizzare l’andamento del PSNR e dell’MSE al variare del numero di iterazioni.
    Facendo variare il parametro di regolarizzazione λ, analizzare come questo influenza le prestazioni del
        metodo analizzando le immagini. 
    Scegliere λ con il metodo di discrepanza.
    Scegliere λ attraverso test sperimentali come il valore che minimizza il valore del PSNR. Confrontare
    il valore ottenuto con quella della massima discrepanza.
"""

# Regolarizzazione
# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m, n))
    Ax = A(x, K)
    return 0.5 * np.sum(np.square(Ax - y)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m, n)
    ATAx = AT(A(x,K),K)
    d = ATAx - ATy
    return d.reshape(m * n) + Lx

x0 = y.reshape(m*n)
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

"""Exercise 1.4. Testare i punti precedenti su due immagini in scala di grigio con caratteristiche differenti (per
esempio, un’immagine tipo fotografico e una ottenuta con uno strumento differente, microscopio o altro).
Degradare le nuove immagini applicando, mediante le funzioni gaussian kernel(), psf fft(), l’operatore
di blur con parametri:
    σ = 0,5 dimensione del kernel 7 ×7 e 9 ×9
    σ = 1,3 dimensione del kernel 5 ×5
    Aggiungendo rumore gaussiano con deviazione standard nell’ intervallo (0,0.05].
"""
#analogo a prima cambia solo i seguenti parametri...

#sigma1 = 0.5
#k1 = gaussian_kernel(7,3)

#k2 = gaussian_kernel(9,3)

#sigma2 = 1.3
#k = gaussian_kernel(5,3)

#noise = np.random.normal(0,0,5) * sigma

#############################################################################################à
"""pt. 2 (30/11/2023)
    """
    
#modifico lambda x massimizare funzione: ad esempio vado ad analizzare quale PSNR 
# di lambda è il più grande
x0 = y.reshape(m*n)
lambdas = [0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03] 
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

###############################################################################
#from utils_SR import psf_fft, A, AT, gaussian_kernel  # su file es72.py
