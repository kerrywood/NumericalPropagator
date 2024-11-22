import numpy as np
import scipy.interpolate
import astropy.time
import astropy.units as u

# -----------------------------------------------------------------------------------------------------
class body_position:
    '''
    body_interp( astropy.coordinates.get_moon,  jdnow, jdnow+14 )
    body_interp( astropy.coordinates.get_sun, jdnow, jdnow+14 )

    remember that Poliastro calculates perturbations based off the orbit epoch (tofs), so
    t = 10 is 10 seconds from the orbit epoch
    you need to calculate the position of perturbations based on the orbit epoch and offsets

    ---> this just makes it more efficient

    for celestial bodies, we can use an approximation:
        Lagrange the ephemeris on 10 minute intervals,
        when requested, cache values on 1 second chunks
        if the value is in the cache, just return it
        if not, get it, and cache it.
    '''
    def __init__(self, 
                 body_fun : callable,
                 D1       : astropy.time.Time,
                 D2       : astropy.time.Time,
                 spacing=600):
        self._D1         = D1
        self._D2         = D2
        self._adates     = astropy.time.Time( np.arange( D1.jd, D2.jd, spacing/86400.), format='jd' )
        eph              = body_fun(self._adates).cartesian.xyz.to_value(u.km).T
        self.offset_secs =  86400 * (self._adates.jd - self._adates[0].jd) 
        self.eph         = eph.T
        self.max_offset  = np.max( self.offset_secs )
        self.N           = len( self.offset_secs )
        self.cache       = {}
        self.hits        = 0
        self.cache_hits  = 0
        self.spacing     = spacing
        self.body_fun    = body_fun
        self.interp      = scipy.interpolate.interp1d( self.offset_secs, self.eph, assume_sorted=True )
        #self.interp      = scipy.interpolate.interp1d( self.offset_secs, self.eph, kind='cubic', assume_sorted=True )
        print('Initialized body_interp: {} total points, {} second spacing, {} max offset, {} function'.format(
            self.N, self.spacing, self.max_offset, body_fun.__name__))

    def __call__( self, sec):
        if sec > self.max_offset: 
            print('error with {}, {} is beyond max offset of {}'.format( 
                self.body_fun.__name__,
                sec, self.max_offset)  )
            return 0.0

        # for now, round values to a second... this improves cache hits at the expense of accuracy
        key = np.round(sec)
        if key in self.cache: return self.cache[ key ]
        rv = self.interp( key )
        self.cache[ key ] = rv
        return rv


# -----------------------------------------------------------------------------------------------------
# old 'get_moon' routines are gone, and replaced with a parameterized function...
def moon_callable( X ): return astropy.coordinates.get_body('moon', X )
def sun_callable( X ) : return astropy.coordinates.get_body('sun' , X )
def get_moon( D1, D2 ) : return body_position( moon_callable, D1, D2 )
def get_sun( D1, D2 ) : return body_position( sun_callable, D1, D2 )

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import astropy.coordinates
    import astropy.units as u
    ST = astropy.time.Time.now()
    ET = astropy.time.Time.now() + 1 * u.day
    moon =get_moon( ST, ET )