#!/usr/bin/env python
# coding: utf-8

# # SpaceX Epehmerides Test

# In[ ]:


from datetime import datetime, timedelta
import numpy as np
import json
import astropy.time
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import DOP853, solve_ivp

import perturbations


truth_eph             = pd.read_csv('./test/spacex_test.csv')
truth_eph['time_dt']  = pd.to_datetime( truth_eph['time'] )
epoch                 = truth_eph.iloc[0]['time_dt']
truth_eph['tof']      = truth_eph['time_dt'].apply( lambda T: (T-epoch).total_seconds() )
total_tof             = ( truth_eph.iloc[-1]['time_dt'] - epoch ).total_seconds()
print('I see {:5.2f} hours , {} seconds of ephemeris'.format( total_tof/(60*60), total_tof ) )


# find the windows of the ephemeris
D1 = astropy.time.Time(truth_eph['time_dt'].iloc[0])
D2 = astropy.time.Time(truth_eph['time_dt'].iloc[-1])

# get the initial state vector
init_sv = truth_eph.iloc[0][ ['x','y','z','dx','dy','dz'] ].values
print('Initial state vector: {}'.format( init_sv ) )
print('\tP : {}'.format( init_sv[:3] ) )
print('\tV : {}'.format( init_sv[3:] ) )

# perturbations holder
P_twobody = perturbations.perturbations( D1, D2 )
P_j2      = perturbations.perturbations( D1, D2 )
P_j2.addJ2()
P_j2.addAtmosphereExponential( C_D=10 )

#P.addAll()
#P.addMoon()
#P.addJ2()
#P.addJ3()
# P.addAtmosphereExponential()

result_two = solve_ivp(
    P_twobody,
    (0, total_tof),
    init_sv,
    args=(),
    rtol=1e-10,
    atol=1e-12,
    method=DOP853,
    dense_output=True,
    events=None,
)

result_j2 = solve_ivp(
    P_j2,
    (0, total_tof),
    init_sv,
    args=(),
    rtol=1e-10,
    atol=1e-12,
    method=DOP853,
    dense_output=True,
    events=None,
)



err_twobody = [] 
err_j2      = []

for i,R in truth_eph.iterrows():
    teph = R[['x','y','z','dx','dy','dz']].values 
    ivp_two = result_two.sol( R['tof'] )
    ivp_j2  = result_j2.sol( R['tof'] )

    err_twobody.append ( ( R['tof'], np.linalg.norm( ivp_two - teph ) ) )
    err_j2.append ( ( R['tof'], np.linalg.norm( ivp_j2 - teph ) ) )


for two,j2 in zip( err_twobody, err_j2 ):
    print('{:8.3f} {:8.3f} {:8.3f} '.format( two[0]/(60*60), two[1], j2[1] ) )
