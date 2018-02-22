# Replicate 'EASI GMM.do'
from collections import OrderedDict
import pandas as pd

# * Tricks with Hicks: The EASI demand system
# * Arthur Lewbel and Krishna Pendakur
# * 2008, American Economic Review
#
# * This code is written by Krishna Pendakur
# * Suggested citation:  "Pendakur, Krishna. 2015.  "EASI GMM code for Stata".
# * available at www.sfu.ca/~pendakur
# * keywords:  Stata, GMM, system, multiple-equation, demand, EASI
#
# * Herein, find Stata code to estimate a demand system with J equations, J
# * prices, ndem demographic characteristics and npowers powers of implicit
# * utility. Because Stata's GMM routine can only handle about 100 parameters,
# * this will bomb if you have J or ndem very large. In that case, use "EASI
# * GMM moment evaluator.do", which is uglier, but gets the job done. For small
# * demand systems, "EASI GMM.do" and "EASI GMM moment evaluator.do" yield
# * identical results.
#
# * use_D=0 sets the matrix D (zy interactions) to zero.
# * use_B=0 sets the matrix B (py interactions) to zero.
# * use_Az=0 sets the matrices A_l (pz interactions) to zero.

data = pd.read_stata('../easi-stata/hixdata.dta')

# * set number of equations and prices and demographic characteristics
J = 8
ndem = 5
npowers = 5

use_D = False
use_B = False
use_Az = False

data['one'] = 1

# * data labeling conventions:
# * budget shares: s1 to sneq
# * prices: p1 to nprice
# * implicit utility: y, or related names
# * demographic characteristics: z1 to zTdem
data['w1'] = data['sfoodh']
data['w2'] = data['sfoodr']
data['w3'] = data['srent']
data['w4'] = data['soper']
data['w5'] = data['sfurn']
data['w6'] = data['scloth']
data['w7'] = data['stranop']
data['w8'] = data['srecr']
data['w9'] = data['spers']

data['p1'] = data['pfoodh']
data['p2'] = data['pfoodr']
data['p3'] = data['prent']
data['p4'] = data['poper']
data['p5'] = data['pfurn']
data['p6'] = data['pcloth']
data['p7'] = data['ptranop']
data['p8'] = data['precr']
data['p9'] = data['ppers']

# * normalised prices are what enter the demand system
# * generate normalised prices, backup prices (they get deleted), and Ap
for j in range(1, J):
    data['np%d' % j] = data['p%d' % j] - data['p%d' % J]

# * list demographic characteristics: fill them in, and add them to zlist below
data['z1'] = data['age']
data['z2'] = data['hsex']
data['z3'] = data['carown']
data['z4'] = data['time']
data['z5'] = data['tran']

zlist = ' '.join(['z1', 'z2', 'z3', 'z4', 'z5'])

Azlist = zlist if use_Az else ''

# * make y_stone=x-p'w, and gross instrument, y_tilda=x-p'w^bar
data['x'] = data['log_y']
for r in range(1, npowers + 1):
    data['x%d' % r] = data['x'].pow(r)
data['y_stone'] = data['x']
data['y_tilda'] = data['x']
for j in range(1, J + 1):
    data['mean_w%d' % j] = data['w%d' % j].mean()
    data['y_tilda'] -= data['mean_w%d' % j] * data['p%d' % j]
    data['y_stone'] -= data['w%d' % j] * data['p%d' % j]

xzlist = []
npzlist = []
for var in zlist.split(' '):
    print(var)
    # Interactions of z and z
    name = 'x%s' % var
    data[name] = data['x'] * data[var]
    xzlist.append(name)

    # Interactions of np and z
    for j in range(1, J):
        name = 'np%d%s' % (j, var)
        data[name] = data['np%d' % j] * data[var]
        npzlist.append(name)

xzlist = ' '.join(xzlist)
npzlist = ' '.join(npzlist)

print(xzlist, npzlist, sep='\n')

# Interactions of np and x
npx = []
for j in range(1, J):
    name = 'np%dx' % j
    data[name] = data['np%d' % j] * data['x']
    npx.append(name)

npx = ' '.join(npx)

print(npx)

# * make global macros for x_stone and y
x_stone = '( x'
for j in range(1, J + 1):
    x_stone += ' - (p%d*w%d)' % (j, j)
x_stone += ' )'

print(x_stone)

y = '( y_stone '
for j in range(1, J):
    y += ' + 0.5*(np%d)^2*{A%d%d: one %s}' % (j, j, j, Azlist)
    for k in range(j + 1, J):
        y += ' + np%d*np%d*{A%d%d: one %s}' % (j, k, j, k, Azlist)
y += ' )'

print(y)

if use_B:
    denom = '(1 '
    for j in range(1, J):
        denom += '- 0.5*(np%d)^2*{B%d%d}' % (j, j, j)
        for k in range(j + 1, J):
            denom += 'np%d*np%d*{B%d%d}' % (j, k, j, k)
    denom += ' )'
    y = '(%s / %s)' % (y, denom)

# * display y; note that Stata GMM only wants the variable lists in GMM
# * parameter vectors on the first occurrence
y_first = y

print(y_first)

if use_Az:
    y = y.replace(' one ' + ' '.join(zlist), '')
else:
    y = y.replace(' one ', '')

print(y)


eq = OrderedDict()
eqlist = ''
# * make equations
for j in range(1, J):
    eq[j] = '(w%d - {b%d0} - {b%d1}*%s' % (j, j, j, y_first if j == 1 else y)

    for r in range(2, npowers + 2):
        eq[j] += ' - {b%d%d}*%s^%d' % (j, r, y, r)

    eq[j] += '- {C%d:%s}' % (j, zlist)

    if use_D:
        eq[j] += '- {D%d:%s}*%s' % (j, zlist, y)

    for k in range(1, J):
        if k < j:
            eq[j] += '- {A%d%d:}*np%d' % (k, j, k)
            if use_B:
                eq[j] += ' - {B%d%d:}*np%d*%s' % (k, j, k, y)
        else:
            eq[j] += '- {A%d%d:}*np%d' % (j, k, k)
            if use_B:
                eq[j] += ' - {B%d%d:}*np%d*%s' % (j, k, k, y)
    eq[j] += ')'

eqlist = ' '.join(eq.values())

print(eq)

eq[5]

# * create instruments and instrument list
instlist = ['x%d' % r for r in range(1, npowers + 1)]
instlist.append(zlist)
if use_D:
    instlist.append(xzlist)
instlist.extend(['np%d' % j for j in range(1, J)])
if use_Az:
    instlist.append(npzlist)
if use_B:
    instlist.append(npx)

instlist = ' '.join(instlist)

print(instlist)

# gmm `eqlist', inst(`instlist') winitial(unadjusted, independent)
#   quickderivatives
