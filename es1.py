"""
1 MACHINE PRECISION
Exercise 1.1. Machine epsilon or machine precision is an upper bound on the relative approximation error
due to rounding in floating point arithmetic. Execute the following code
    import sys
    help(sys.float_info)
    print(sys.float_info)
• A) understand the meaning of max, max exp and max 10 exp.
• B) Write a code to compute the machine precision ϵ in (float) default precision with a while construct.
Compute also the mantissa digits number.
• C) Use NumPy and exploit the functions float16 and float31 in the while statement and see the
differences. Check the result of np.finfo(float).eps.
"""
#A)

import sys
#help(sys.float_info)
print(sys.float_info)

""" 
Result:
sys.float_info(max=1.7976931348623157e+308, max_exp=1024, 
max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, 
mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)

max =  maximum representable finite float
max_exp = maximum int e such that radix**(e-1) is representable
max_10_exp = maximum int e such that radix**(e-1) is representable
"""
#B)

epsilon = 1.0
mant_dig = 1
while 1.0 + epsilon / 2.0 > 1.0:
    epsilon /= 2.0
    mant_dig += 1
print("epsilon macchina: ", epsilon)
print("mantissa: ", mant_dig)
""" 
Result:
esattamente come sopra: mant_dig=53, epsilon=2.220446049250313e-16.
l'epsilon lo trovo dividendo per 2 fino a quando il numreo è talmente piccolo
che il calcolatore non riesce più a validare lo statement del while
"""
#C)

import numpy as np

epsilon16 = np.float16(1.0)
mant_dig16 = 1
while np.float16(1.0) + epsilon / np.float16(2.0) > np.float16(1.0):
    epsilon /= np.float16(2.0)
    mant_dig += 1
print("epsilon macchina in float16: ", epsilon16)
print("mantissa in float16: ", mant_dig16)

epsilon32 = np.float32(1.0)
mant_dig32 = 1
while np.float32(1.0) + epsilon / np.float32(2.0) > np.float32(1.0):
    epsilon /= np.float32(2.0)
    mant_dig += 1
print("epsilon macchina in float32: ", epsilon32)
print("mantissa in float32: ", mant_dig32)

""" 
Result:
la precisione di calcolo è aumentata subito epsilon: 1.0 e mantissa: 1
"""
##############
""" 
2 PLOT OF A FUNCTION
Exercise 2.1. Matplotlib is a plotting library for the Python programming language and its numerical
mathematics extension NumPy. Create a figure combining together the cosine and sine curves, on the
domain [0, 10]:
    • add a legend
    • add a title
    • change the default colors
"""
import matplotlib.pyplot as plt
import math as mt

linspace = np.linspace(0, 10);                                              # spaziatura 0 to 10
plt.subplots(constrained_layout = True)[1].secondary_xaxis(0.5);            # abilitazione sovrapposiz grafi e aggiunta assale x in mezzo
plt.plot(linspace, np.sin(linspace), color='green')                         
plt.plot(linspace, np.cos(linspace), color='red')
plt.legend(['Sin(x)', 'Cos(x)'])                                            
plt.title('Sin(x) e Cos(x) con 0 <= x <= 10')                               
plt.show()

""" 
Exercise 2.2. The Fibonacci sequence is a sequence in which each number is the sum of the two preceding
ones and it is formally defined as:
  {  F1 = F2 = 1
  {   Fn = Fn-1 + Fn-2 n>2
• A) Write a script that, given an input number n, computes the number Fn of the Fibonacci sequence.
• B) Write a code computing, for a natural number k, the ratio rk = Fk+1/Fk, , where Fk are the Fibonacci
    numbers.
• C) Verify that, for a large k, {{rk}}k converges to the value φ=1+√5/2
• D) Create a plot of the error (with respect to φ)
"""
#A)
print("n?")
n = input()
n = int(n)
if n <= 0:
    cont = 0
elif n <= 1:
    cont = 1
else:
    a, b = 0, 1
    cont = 2
    while a + b < n:
        b += a
        a = b - a
        cont += 1
print("Fibonacci di ", n, ": ", cont)   #fibonacci risult.

#B)
k = n
if k <= 0:
    print("0") 
a, b = 0, 1
for i in range(k):
    b += a
    a = b - a
rk = b / a
print("r(k): ",rk) #golden ratio

#C)
phi = (1.0 + 5 ** 0.5) / 2.0  #golden ratio = 1.6180339887...
print(abs(rk - phi) / phi)    
""" 
Result:
per numeri grandi, tipo mille, l'ultimo print da 0.0, ciò significa che va a 
convergere diminuendo l'errore relativo
"""
#D)
arange = np.arange(n)                       # lista di n valori, da 0 a n
plt.plot(arange, [rk for i in arange])      # grafico con i valori arange nelle ascisse 
                                            # e l'errore relativo al valore i-esimo di arange rk
plt.legend(['relative error'])
plt.show()
""" 
l'asse y l'errore relativo di rk e, al crescere di quest'ultimo, rk viene approssimato con 
piu' precisione, e il relativo errore decresce, come si può vedere dal grafico
"""