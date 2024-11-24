import json
import MEMEreader
D = MEMEreader.openMEME('../singleton/spacex/MEME_47798_STARLINK-2379_3171301_Operational_1415710920_UNCLASSIFIED.txt.gz')
eph = np.vstack( D['state_vecs'] )

out = pd.DataFrame()
out['time'] = D['dates']
out['x'] = eph[:,0]
out['y'] = eph[:,1]
out['z'] = eph[:,2]
out['dx'] = eph[:,3]
out['dy'] = eph[:,4]
out['dz'] = eph[:,5]

out.to_csv('./test/spacex_test.csv', index=None )