import numpy as np
import math as mt 
import random as rd 
import matplotlib.pyplot as plt

#UTILITY#

#tipologia dati
# a = 3 int
# b = 2.5 float
# c = 'ciao' stringa
# d = True booleano (True/False)
# e = 2 + 3j complex

#operatori aritmetici particolari
# ** elevamento potenza
# // divisione intera

#liste
# l.append per mettere elemento all'inizio
#
#condizione
# elif al posto di else if

#ciclo
# for i (indice )in v (vettore)
# 
# range(n) = lista da 0 a n-1

#funzioni
# def funzione(): 

#numpy
# np.array() creaz array/matrici
# .size num elem
# .shape num righe, num colonne
# .ndim num dimensioni
# np.repeat(x,2) vett con elem 2 volte
# np.arange(1,11,1) vett da 1 a 10
# np.linespace(0,50,6) vett di 6 equispaziati da 0 a 50
# np.sort() ordina el in modo cresc
# np.where(x >= 3) posiz da cui vale condiz
# np.concatenate() concat matr
# np.transponse() trasposta
# np.flatten() data matr da valore di essa appiattita
# A[i,:] per estrarre riga
# A[i:j,:] per estrarre righe consec
# analogo per colonne A[:,i:j]
# A[i1:i2,j1:j2] sottomatrice

#matplotlib
# plt.plot(x ascisse, y ordinate, color='green' colre grafo, marker='o' rappresenta punti, linestyle='dashed' tipo di linea)
# plt.xlabel('')
# plt.ylabel('')
# plt.title('')
# plt.show()
# plt.plot(,,'1') disegna grafo nel grafo
# ...'2') secondo grafo nello stesso
# plt.legend([''black','red']) colori 2 grafi
#
# fig, ax = plt.subplots(nrows=2, ncols=2) er disegnare più plot nella stessa schermata si usa la funzione
# plt.subplots(nrows=rnum, ncols=cnum), dove rnum rappresenta
#il numero di plot che si vuole visualizzare in riga, mentre cnum
#rappresenta il numero di plot che si vuole visualizzare in colonna
# ax[0,0].plot(x, y, ’ko’)
# ax[0,1].plot(x, y, ’k-’)
# ax[1,0].plot(x, y, ’k-o’)
# ax[1,1].plot(x, y, ’k--o’)
#
# Per disegnare una retta si usa il comando axline(xy1, xy2=None,slope=None)
#xy1, xy2 punti per cui passa la retta.
#
#slope coefficiente angolare retta (se non si specifica xy2)
# plt.text(0,0.8, ’pto di min’, horizontalalignment=’center’) label sul grafico
# np.random.choice(v, 10, replace=True, p=(0.7 , 0.2 , 0.1Dato un vettore v la funzione random.choice() estrae da v (conreinserimento) n valori
# La funzione random.normal() prende in input
# x = random.normal(loc=0.0, scale=1.0, size=10)la media,la deviazione standard,la lunghezza del vettore di output restituisce un vettore che ha dimensione scelta e contiene elementi estratti
#con una distribuzione Gaussiana con media e deviazione assegnate
#
# plt.hist(x, bins=100) Per disegnare un istogramma si utilizza la funzione hist() che prende come input un vettore (del quale plottare le frequenze) e un parametro opzionale bins, che indica in quanti intervalli dividere 
# i valori dell’array in input
#
# bar() per rappresentare dei grafici a barre
#piechart() per rappresentare dei grafici a torta
#Prima è necessario contare le occorrenze di ogni possibile valore.
#> x = np.array(("a" , "a" , "a" , "b" , "a" , "b" ))
#> unique = np.unique(x)
#> count = [np.sum(x==el) for el in unique]
#> plt.bar(unique,count)
#> plt.pie(count,labels=unique)
#
#Il boxplot(), risulta utile per visualizzare il range dei dati.
#> x = np.array((3, 3, 3, 2, 1, 4, 6, 2, 4, 1, 6, 4, 2, 1, 5, 4
#> plt.boxplot(x)

##ESAMPI

x = np.arange(11) #vett di grandezza 5 di int 0-4
y = np.arange(-5,6) #x DEVE essere stessa dim x stare sullo steso grafo
z = y**2
plt.plot(x, '--', color='green') #vet tipo linea, colore
plt.plot(x,z,'-o',color='black')
plt.show()

fig, ax=plt.subplots(nrows = 2, ncols = 2)
ax[0,0].plot(x,z)
ax[0,1].plot(y)
plt.show()

x1 = np.linspace(-1,10,100) #linspace(0,1,50) vett 0,1 con 50 num 
g = np.exp(x1)
fig, bu=plt.subplots(nrows = 2, ncols = 2)
bu[0,0].plot(x1, g)
bu[0,1].semilogy(x1,g)
bu[1,0].semilogx(x1,g)
bu[1,1].loglog(x1,g)
plt.show()