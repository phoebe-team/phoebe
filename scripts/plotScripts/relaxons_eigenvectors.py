import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

f = h5py.File('el_relaxons_eigenvectors.hdf5', 'r')
# theta0 special eigenvector
#eigenvector = f['theta0'][:,0]
# third relaxon in order of relaxation time, first band
eigenvector = f['relaxonEigenvectors_2'][:,0]

# unpack the json file
kpts = np.array(f['wavevectorCoordinatesCartesianWS'])
# plot aesthetics
cm = plt.get_cmap('RdBu_r')
fig = plt.figure(figsize=(4.5,3.25))
ax = fig.add_subplot(projection='3d')

ax.set_aspect('equal')
ax.autoscale(enable=False)

# plot the data
p = ax.scatter(kpts[:,0], kpts[:,1], kpts[:,2], c=eigenvector, marker='o',s=2, cmap=cm)

cbar = fig.colorbar(p,shrink=0.6)
p.set_clim(-0.05,0.05)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.set_xlabel("kx [1/$\AA$]")
ax.set_ylabel("ky [1/$\AA$]")
ax.set_zlabel("kz [1/$\AA$]")
cbar.ax.set_title('$\\theta^\\alpha_\\nu$',pad=10)
ax.view_init(elev=15, azim=45, roll=0)
plt.savefig('eig.png',dpi=250)
