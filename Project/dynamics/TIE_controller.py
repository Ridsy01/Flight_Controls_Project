import numpy as np

TIE_speed = 2.

def TIEcontrol(X_state, TIE_state):

    dn = (X_state[0][0] - TIE_state[0][0])/20
    de = (X_state[1][0] - TIE_state[1][0])/2
    dd = (X_state[2][0] - TIE_state[2][0])/2

    if dn > 0: TIE_state[0][0] = TIE_speed * np.sqrt(dn)
    else: TIE_state[0][0] = - TIE_speed * np.sqrt(abs(dn))

    if de > 0: TIE_state[1][0] = TIE_speed * np.sqrt(de)
    else: TIE_state[1][0] = - TIE_speed * np.sqrt(abs(de))

    if dd > 0: TIE_state[2][0] = TIE_speed * np.sqrt(dd)
    else: TIE_state[2][0] = - TIE_speed * np.sqrt(abs(dd))

    return TIE_state
