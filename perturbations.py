from datetime import datetime, timedelta
import numpy as np
import astropy.time
import astropy.units as u

from scipy.integrate import DOP853, solve_ivp
from poliastro.core.perturbations import third_body, J2_perturbation, J3_perturbation, radiation_pressure, atmospheric_drag_exponential
from poliastro.bodies import Earth, Moon, Sun
from poliastro.ephem import build_ephem_interpolant
from poliastro.constants import H0_earth, Wdivc_sun, rho0_earth
from poliastro.twobody import Orbit
from poliastro.core.propagation import func_twobody
import scipy.interpolate

import body_position

# -----------------------------------------------------------------------------------------------------
# constants : our canonical units are km and s
J2     = Earth.J2.value
J3     = Earth.J3.value
EarthR = Earth.R.to(u.km).value
EarthK = Earth.k.to(u.km ** 3 / u.s ** 2).value
MoonK  = Moon.k.to(u.km**3 / u.s**2).value
SunK   = Sun.k.to(u.km**3 / u.s**2).value
WdivcC = Wdivc_sun.to_value( u.kg * u.km / u.s**2)
rho0C  = rho0_earth.to(u.kg / u.km**3).value
H0C    = H0_earth.to(u.km).value  # km

# -----------------------------------------------------------------------------------------------------
class perturbations: 
    def __init__( self, 
                 D1         : astropy.time.Time,
                 D2         : astropy.time.Time,
                 C_R        : float = 1.0, 
                 AoverM     : float = 1e-6, 
                 C_D        : float = 1.0, 
                 K          : float = EarthK,
                 BUFFER     : float = 5.):
        self.forces     = {}
        self.twobody    = func_twobody
        self._atime1    = D1
        self._atime2    = D2
        self._jd1       = self._atime1.jd
        self._jd2       = self._atime2.jd
        self.moon_cache = None
        self.sun_cache  = None
        self.C_R        = C_R    # only used when SRP is active
        self.AoverM     = AoverM # only used with SRP
        self.C_D        = C_D    # for atmo drag
        self.K          = K
        self._buffer    = BUFFER

    @property
    def max_offset( self ):
        return (self._atime2 - self._atime1).to_value(u.s)

    def clearForces( self ): self.forces.clear()
    def clearJ2( self ) : self.forces.pop('j2', None)
    
    def addJ2( self ): self.forces['j2'] = lambda *X: J2_perturbation( *X, J2, EarthR )
    def addJ3( self ): self.forces['j3'] = lambda *X: J3_perturbation( *X, J3, EarthR )

    def setSunCache( self ):
        if self.sun_cache == None: 
            self.sun_cache = body_position.get_sun( self._atime1, self._atime2 + self._buffer )

    def setMoonCache( self ):
        if self.moon_cache == None: 
            self.moon_cache = body_position.get_moon( self._atime1, self._atime2 + self._buffer )

    def addMoon( self ):
        self.setMoonCache()
        self.forces['moon'] = lambda *X: third_body( *X, MoonK, self.moon_cache )

    def addSun( self ):
        self.setSunCache()
        self.forces['sun'] = lambda *X: third_body( *X, SunK, self.sun_cache)

    def addSRP( self ):
        self.setSunCache()
        # radiation_pressure(t0, state, k, R, C_R, A_over_m, Wdivc_s, star)
        self.forces['srp'] = lambda *X: radiation_pressure( *X, 
                EarthR, self.C_R, self.AoverM, WdivcC, self.sun_cache )

    def addAtmosphereExponential( self, C_D=1.0, AoverM=1e-6 ):
        self.C_D    = C_D
        self.AoverM = AoverM
        # radiation_pressure(t0, state, k, R, C_R, A_over_m, Wdivc_s, star)
        self.forces['drag'] = lambda *X: atmospheric_drag_exponential( 
            *X, EarthR, self.C_D, self.AoverM, H0C, rho0C )

    def addAll( self ):
        self.addMoon()
        self.addSun()
        self.addJ2()
        self.addJ3()
        self.addSRP()
        self.addAtmosphereExponential()

    def __call__( self, *args ) :
        # all perturbation functions start with the first three params
        #   perturb( t0, state, k)
        #       t0    : time since epoch
        #       state : six element (P,V) vector
        #       k     : Earth K value
        #   lots ignore K entirely, not sure why this is the prototype
        assert len(args) == 2
        rv = self.twobody( *args , self.K )
        for k in self.forces: 
            # try: rv[3:] += self.forces[k]( *args, self.K )
            try: rv[3:] += self.forces[k]( args[0], args[1], self.K )
            except Exception as e:
                print('error with {} {} {}'.format(k,args,e))
                assert 1 == 2
        return rv

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import astropy.coordinates
    import astropy.units as u
    ST = astropy.time.Time.now()
    ET = astropy.time.Time.now() + 1 * u.day
    P = perturbations( ST, ET )
    FAKE_STATE = np.array( [7000.,0.,0.,0.,0.,0.], dtype=np.float64 )
    P.addAll()
    for X in np.arange(0, P.max_offset, 60 ):
        print( X, P(X, FAKE_STATE) )