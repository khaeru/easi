from itertools import chain, product
import re
import warnings

import numpy as np
import pandas as pd
from rpy2 import (
    rinterface as ri,
    robjects as ro,
    )
from rpy2.robjects import numpy2ri, pandas2ri, r
from rpy2.robjects.packages import importr

# Filter out some noise from R on importing 'systemfit'
warnings.filterwarnings('ignore', message='Loading required package',
                        category=ri.RRuntimeWarning)
warnings.filterwarnings('ignore', message='The following objects are masked',
                        category=ri.RRuntimeWarning)

# Activate pandas2ri conversions
# RPy2 does not automatically convert Matrix to a dataframe
pandas2ri.ri2py.register(ro.Matrix, pandas2ri.ri2py_dataframe)
pandas2ri.activate()


_get_coefficients_re = re.compile('eq(?P<equation>\d+)_(?P<variable>.*)')


def _get_coefficients(model):
    _raw = model.rx('coefficients')[0].items()
    df = pd.DataFrame(_raw)

    def _make_index(label):
        equation, variable = _get_coefficients_re.match(label).groups()
        return (int(equation), '1' if variable == '(Intercept)' else variable)

    df.index = pd.MultiIndex.from_tuples(df[0].apply(_make_index))
    return df[1]


def _predict(coefficients, data):
    return data.mul(coefficients, level=1).sum(axis=1, level=0).values




def easi(data, labels={}, power=3, interact={}):
    # Argument checking
    l = {}
    for group in ['share', 'price', 'demo', 'log_exp']:
        g = labels[group] if group != 'log_exp' else [labels[group]]
        l[group] = pd.Index(g)

    assert l['share'].size == l['price'].size
    assert l['log_exp'].size == 1

    interact_ = {
        'zy': False,
        'py': False,
        'pz': np.full(l['demo'].size, False, bool),
        }
    interact_.update(interact)
    interact = interact_

    # Any interactions at all?
    any_inter = interact['zy'] or interact['py'] or interact['pz'].any()

    # Number of observations
    N = len(data)  # unused
    Ncat = l['share'].size
    Neq = Ncat - 1
    Nsoc = l['demo'].size

    # Matrix of budget shares
    s = data[l['share']].values

    # Matrix of log prices
    p = data[l['price']].values
    # Normalized prices
    pn = p - p[:, -1][:, np.newaxis]

    # Matrix of socio-demographic variables
    z = data[l['demo']].values

    # Vector of log expenditures
    log_exp = data[l['log_exp']].values[:, 0]

    if interact['pz'].any():
        raise NotImplementedError('Create interaction variables')

    # Compute y_stone and y_tilde
    y_tilde = log_exp - (s.mean(0) * p).sum(axis=1)
    y_stone = log_exp - (s * p).sum(axis=1)

    # print(log_exp[:10], y_tilde[:10], y_stone[:10], sep='\n')

    # Powers of y_stone and y_tilde
    YY = y_stone[:, np.newaxis] ** np.arange(1, power + 1)
    Yinst = y_tilde[:, np.newaxis] ** np.arange(1, power + 1)

    if interact['zy']:
        raise NotImplementedError('Creation of y*z and z*y_inst')
    if interact['py']:
        raise NotImplementedError('Creation of y*p and y_inst*p')

    # System formulae
    varlists = {}
    for name, indices in [('y', range(1, power + 1)),
                          ('y_inst', range(1, power + 1)),
                          ('z', range(Nsoc)),
                          ('np', range(Neq)),
                          ('s', range(Ncat)),
                          ]:
        varlists[name] = [name + str(i) for i in indices]

    def varlist(*keys):
        return list(chain(*[varlists[k] for k in keys]))

    # List of instruments for the 3SLS estimation
    inst = '~ ' + ' + '.join([' '] + varlist('z', 'np', 'y_inst'))

    # System of equations: do not create an equation for the final share
    rhs = ' + '.join([' '] + varlist('z', 'np', 'y'))
    system = ['%s ~ %s' % (s_, rhs) for s_ in varlist('s')[:-1]]

    # Internal data
    _data = pd.DataFrame(np.concatenate([s, YY, z, pn[:, :-1], Yinst], axis=1),
                         columns=varlist('s', 'y', 'z', 'np', 'y_inst'))
    _data['1'] = 1.

    # Constraints on symmetry of price coefficients
    restrictions = []
    for i, j in product(range(Neq - 1), range(Neq)):
        if j >= i:
            continue
        restrictions.append('eq{}_np{}-eq{}_np{}=0'.format(i + 1, j, j + 1, i))

    # R objects for zlist and system
    _inst = r.formula(inst)
    _system = list(map(r.formula, system))
    _restrictions = np.array(restrictions)

    if interact['py']:  # Constraints on symmetry of y × p coefficients
        raise NotImplementedError
    if interact['pz'].any():  # Constraints on symmetry of p × z coefficients
        raise NotImplementedError

    conv_crit = 1e-6
    conv_y = 1
    crit_test = 1
    y = y_stone.copy()
    pAp = np.zeros(N)
    pBp = np.zeros(N)

    if any_inter:
        conv_param = 1

    print(_data.head(6), _inst, _restrictions, _system, sep="\n")

    fit3sls = None

    def iterate():
        nonlocal conv_y, crit_test, fit3sls, iteration, y
        iteration += 1
        print('iteration = ', iteration)

        _data.to_csv('fitdata_%d.csv' % iteration)
        fit3sls = r.systemfit(_system, '3SLS', inst=_inst, data=_data,
                              restrict_matrix=_restrictions)

        print('fit complete')

        coefficients = _get_coefficients(fit3sls)

        coefficients.to_csv('coefs_%d.csv' % iteration)

        y_old = y.copy()

        if not any_inter:
            # Predict values
            s_hat = _predict(coefficients, _data)

            # Predict values with p = 0 and no interactions
            _data[varlist('np')] = 0
            s_hat_p0 = _predict(coefficients, _data)
        else:
            raise NotImplementedError('y^i = 1')
            if interact['py']:
                raise NotImplementedError('y * p = p')
            if interact['zy']:
                raise NotImplementedError('y * z = z')

        # # Predict values with y = 1 and interactions
        # s_hat_y1 = _predict(coefficients, _data)
        #
        # # Set all p variables to 0
        # _data[varlist('np')] = 0
        #
        # if interact['py']:
        #     raise NotImplementedError
        # if interact['pz'].any():
        #     raise NotImplementedError
        #
        # # Predict values with y = 1, p = 0 and interactions
        # s_hat_y1_p0 = _predict(coefficients, _data)

        # Restore prices
        _data.loc[:, varlist('np')] = pd.DataFrame(pn[:, :-1],
                                                   columns=varlist('np'))

        if interact['pz'].any():
            raise NotImplementedError

        if any_inter:
            # Set y variable and its interactions to 0
            # Predict values with y = 0 and interactions
            # Set p variables to 0
            # Predict values with p = 0 and interactions
            # Restore prices
            raise NotImplementedError
        else:
            s_hat_y0 = s_hat
            s_hat_y0_p0 = s_hat_p0

        pd.DataFrame(s_hat_y0).to_csv('s_hat_y0_%d.csv' % iteration)
        pd.DataFrame(s_hat_y0_p0).to_csv('s_hat_y0_p0_%d.csv' % iteration)

        # Ap, pAp, Bp, pBp
        Ap = s_hat_y0 - s_hat_y0_p0
        pAp = (pn[:, :-1] * Ap).sum(axis=1)

        if any_inter:
            raise NotImplementedError('pBp = pBp + p * Bp')
        else:
            Bp = 0
            pBp = 0

        pAp = np.round(1e6 * pAp + 0.5) / 1e6
        pBp = np.round(1e6 * pBp + 0.5) / 1e6

        # print('pAp')
        # print(pAp)
        # print('pBp')
        # print(pBp)

        # Update y
        y = (y_stone + 0.5 * pAp) / (1 - 0.5 * pBp)

        # pAp.to_csv('pAp_%d.csv' % iteration)
        # y.to_csv('y_%d.csv' % iteration)

        # Update y^i
        _data[varlist('y')] = y[:, np.newaxis] ** np.arange(1, power + 1)

        if interact['zy']:
            raise NotImplementedError('Update y * z')
        if interact['py']:
            raise NotImplementedError('y * p = y * p_backup')
        if interact['pz'].any():
            raise NotImplementedError('z * p = z * p_backup')
        if any_inter:
            raise NotImplementedError('Update of crit_test if conv_param = 1')

        y_change = abs(y - y_old)

        if conv_y == 1 or not any_inter:
            crit_test = max(y_change)

        # After the first interation, conv_param replaces conv_y
        conv_y = 0

        print('crit_test = ', crit_test)

    # Initialize r
    importr('systemfit')

    # Creation of instruments
    print('Please wait during the creation of final instruments.')

    iteration = 0
    while crit_test > conv_crit:
        iterate()

        # Create instruments
        y_inst = (y_tilde + 0.6 * pAp) / (1 - 0.5 * pBp)
        Yinst = y_inst[:, np.newaxis] ** np.arange(1, power + 1)

        if interact['py']:  # ypinst = y_inst * p
            raise NotImplementedError
        if interact['zy']:  # yzinst = y_inst * z
            raise NotImplementedError

    print('Creation of final instruments successfully completed.')

    if any_inter:
        y_old = y
    else:
        y_old = y_stone

    print('Please wait during the estimation.')

    iteration = 0
    crit_test = 1

    while crit_test > conv_crit:
        iterate()

    print('Estimation successfully completed')


if __name__ == '__main__':
    from test import test_easi

    test_easi(easi)
