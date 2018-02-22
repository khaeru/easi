# Replicate 'iterated_3sls_without_pz,py,zy.do'

# * Tricks with Hicks: The EASI demand system
# * Arthur Lewbel and Krishna Pendakur
# * 2008, American Economic Review
#
# * Herein, find Stata code to estimate a demand system with neq equations,
# * nprice prices, ndem demographic characteristics and npowers powers of
# * implicit utility
# set more off
# macro drop _all
# use "C:\projects\hixtrix\revision\hixdata.dta", clear

import pandas as pd

data = pd.read_stata('../easi-stata/hixdata.dta')

# * set number of equations and prices and demographic characteristics and
# * convergence criterion
neqminus1 = 7
neq = 8
nprice = 9
ndem = 5
npowers = 5
conv_crit = 1e-6

# * data labeling conventions:
# * budget shares: s1 to sneq
# * prices: p1 to nprice
# * implicit utility: y, or related names
# * demographic characteristics: z1 to zTdem
data['s1'] = data['sfoodh']
data['s2'] = data['sfoodr']
data['s3'] = data['srent']
data['s4'] = data['soper']
data['s5'] = data['sfurn']
data['s6'] = data['scloth']
data['s7'] = data['stranop']
data['s8'] = data['srecr']
data['s9'] = data['spers']

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
for j in range(1, neq + 1):
    data['np%d' % j] = data['p%d' % j] - data['p%d' % nprice]
    data['np%d_backup' % j] = data['np%d' % j]
    data['Ap%d' % j] = 0

data['pAp'] = 0

# * list demographic characteristics: fill them in, and add them to zlist below
data['z1'] = data['age']
data['z2'] = data['hsex']
data['z3'] = data['carown']
data['z4'] = data['time']
data['z5'] = data['tran']

zlist = 'z1 z2 z3 z4 z5'

# * make y_stone=x-p'w, and gross instrument, y_tilda=x-p'w^bar
data['x'] = data['log_y']
data['y_stone'] = data['x']
data['y_tilda'] = data['x']
for num in range(1, nprice + 1):
    data['mean_s%d' % num] = data['s%d' % num].mean()
    data['y_tilda'] -= data['mean_s%d' % num] * data['p%d' % num]
    data['y_stone'] -= data['s%d' % num] * data['p%d' % num]

# * list of functions of (implicit) utility, y: fill them in, and add them to
# * ylist below
# * alternatively, fill ylist and yinstlist with the appropriate variables and
# * instruments
data['y'] = data['y_stone']
data['y_inst'] = data['y_tilda']

ylist = ''
yinstlist = ''

for j in range(1, npowers + 1):
    data['y%d' % j] = data['y'].pow(j)
    data['y%d_inst' % j] = data['y_inst'].pow(j)
    ylist += ' y%d' % j
    yinstlist += ' y%d_inst' % j

# * set up the equations and put them in a list
eq = {}
eqlist = ''
for num in range(1, neq + 1):
    eq[num] = '(s%d %s %s np1-np%d)' % (num, ylist, zlist, neq)
    print(eq[num])
    eqlist += ' %s' % eq[num]

# * create linear constraints and put them in a list, called conlist
constraints = {}
conlist = ''
for j in range(1, neq + 1):
    for k in range(j + 1, neq + 1):
        con_id = '%d%d' % (j, k)
        constraints[con_id] = '[s%d]np%d=[s%d]np%d' % (j, k, k, j)
        conlist += ' ' + con_id

# At this point we're stuck, because there is no 3SLS for Python
