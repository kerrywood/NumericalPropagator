from datetime import datetime, timedelta
import numpy as np
import astropy.time
import astropy.units as u
import pandas as pd
from sgp4.propagation import sgp4 as sgprop
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72, wgs84
import astropy.coordinates
import astropy.time
import astropy.units as u
import logging 
from scipy.integrate import DOP853, solve_ivp

from NumericalPropagator import perturbations

# -----------------------------------------------------------------------------------------------------
class propagator( perturbations.perturbations ):   
    def __init__( self, date1 : datetime = None, date2 : datetime = None ):
        if date1 is None:
            date1 = astropy.time.Time( '2000-1-1' )
        if date2 is None:
            date2 = astropy.time.Time( '2000-1-2' )

        self._d1 = date1
        self._d2 = date2
        super().__init__( date1, date2 )

   #47         self.AoverM     = AoverM # only used with SRP
    #48         self.C_D        = C_D    # for atmo drag
    #49         self.K          = K
    #50         self._buffer    = BUFFER

    def set_dates( self, D1, D2 ):
        super().__init__( date1, date2 )
        return self

    def propagate_sv( self, 
                      epoch : astropy.time.Time,
                      sv : np.array, 
                      dates : list[ astropy.time.Time]  ,
                      ):
        offset_s = ( dates.jd - dates.jd[0] ) * 86400
        assert all( np.diff(offset_s) > 0 ) 
        # make sure we have perturbations in this range
        if not( dates[0] >= self._d1 and dates[-1] <= self._d2 ):
            logging.info('current perturbations do not cover range, resetting')
            super().__init__( dates[0] , dates[1] )
        result = solve_ivp(
                        self,
                        (offset_s[0], offset_s[-1] ),
                        sv,
                        args=(),
                        rtol=1e-10,
                        atol=1e-12,
                        method=DOP853,
                        dense_output=True,
                        events=None,
        )
        if not result.success: 
            return None
        return np.vstack( result.sol(offset_s) ).T


# =====================================================================================================
if __name__ == "__main__":
    P = propagator() 
    P.addJ2()
    P.addAll()
    print(P.getActiveForces())
    epoch = pd.Timestamp.now()
    dates = pd.date_range( epoch, epoch + pd.Timedelta(days=5), freq='5min')
    dates = astropy.time.Time( dates )
    ans = P.propagate_sv( epoch,
                        np.array([7000,0,0,0,7,0]),
                        dates )
    print(ans)

