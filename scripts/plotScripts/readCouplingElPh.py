import numpy as np
import h5py

# read in the file 
f = h5py.File('coupling.elph.phoebe.hdf5', 'r')

# k and q are reconstructed from this below
points = np.array(f["pointsPairsCrystal"])
points = np.reshape(points,(6,-1)).T
numPointsPairs = points.shape[0]

# unfold the points, which are listed in pairs as kx ky kz qx qy qz
ks = points[:,0:3]
qs = points[:,3:6]

# set up band info
bandRange1 = np.array(f["elBandRange1"])
bandRange2 = np.array(f["elBandRange2"])
bandRangePh = np.array(f["phModeRange"])
numBands1 = bandRange1[1] - bandRange1[0] + 1 
numBands2 = bandRange2[1] - bandRange2[0] + 1
numBandsPh = bandRangePh[1] - bandRangePh[0] + 1

couplingMat = np.array(f["elphCouplingMat"])
couplingMat= np.reshape(couplingMat, (numPointsPairs, numBands1, numBands2, numBandsPh))

print(couplingMat)
