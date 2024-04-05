# Save the following to WannierEph.py:
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
np.set_printoptions(threshold=sys.maxsize,linewidth=sys.maxsize,precision=4)

# a few notes here -- how should I use the cell weights? 
# will performance be better if I use JDFTx's Re and Rp? 

# READ IN phoebe components ----------------------------------------------
# load in the R vectors from Phoebe to calculate the phase
filename = "../inputs/16x16x16/cu.phoebe.elph.hdf5"
with h5py.File(filename, "r") as f:
    # get first object name/key; may or may NOT be a group
    cellMapRe = f["elBravaisVectors"][()]
    cellMapRp = f["phBravaisVectors"][()]
    cellWeightsRe = f["elDegeneracies"][()]
    cellWeightsRp = f["phDegeneracies"][()]

# convert the phoebe R vectors to lattice coordinates 
# (if kpts and qpts are in crystal, R vectors should be too) 
unitCell = np.array([[  0.00000000,   1.76692271,   1.76692271 ],
   [ 1.76692271,   0.00000000,   1.76692271 ],
   [ 1.76692271,   1.76692271,   0.00000000 ]])/0.529177

cellMapRe = np.matmul(np.linalg.inv(unitCell), cellMapRe)
cellMapRp = np.matmul(np.linalg.inv(unitCell), cellMapRp)

# READ IN JDFTX components -----------------------------------------------
# Read the MLWF cell map, weights and Hamiltonian:
  =cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(int)
Wwannier = np.fromfile("wannier.mlwfCellWeights")

nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)
# --- Get cell volume, mu and k-point folding from totalE.out:
for line in open('totalE.out'):
    if line.startswith("unit cell volume"):
        Omega = float(line.split()[-1])
    if line.startswith('\tFillingsUpdate:'):
        mu = float(line.split()[2])
    if line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])
# --- Read reduced Wannier Hamiltonian, momenta and expand them:
Hreduced = np.fromfile("wannier.mlwfH").reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
Preduced = np.fromfile("wannier.mlwfP").reshape((kfoldProd,3,nBands,nBands)).swapaxes(2,3)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]
Pwannier = Wwannier[:,None] * Preduced[iReduced]

# Read phonon dispersion relation:
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[0,1,2]).astype(int)
nCellsPh = cellMapPh.shape[0]
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)

print(cellMapRe.shape, cellMap.shape, cellWeightsRe.shape, Wwannier.shape)
print(cellMapRp.shape, cellMapPh.shape, cellWeightsRp.shape, Wwannier.shape)
# (3, 4497) (5441, 3) (4497, 1) (348224,)
# (3, 617) (93, 3) (617, 1) (5441, 8, 8)

# Read e-ph matrix elements
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[0,1,2]).astype(int)
nCellsEph = cellMapEph.shape[0]
# --- Get phonon supercell from phonon.out:
for line in open('phonon.out'):
    tokens = line.split()
    if len(tokens)==5:
        if tokens[0]=='supercell' and tokens[4]=='\\':
            phononSup = np.array([int(token) for token in tokens[1:4]])
prodPhononSup = np.prod(phononSup)
phononSupStride = np.array([phononSup[1]*phononSup[2], phononSup[2], 1])
# --- Read e-ph cell weights:
nAtoms = nModes // 3
cellWeightsEph = np.fromfile("wannier.mlwfCellWeightsPh").reshape((nCellsEph,nBands,nAtoms)).swapaxes(1,2)
cellWeightsEph = np.repeat(cellWeightsEph.reshape((nCellsEph,nAtoms,1,nBands)), 3, axis=2)  # repeat atom weights for 3 directions
cellWeightsEph = cellWeightsEph.reshape((nCellsEph,nModes,nBands))  # coombine nAtoms x 3 into single dimension: nModes
# --- Read, reshape and expand e-ph matrix elements:
iReducedEph = np.dot(np.mod(cellMapEph, phononSup[None,:]), phononSupStride)
HePhReduced = np.fromfile('wannier.mlwfHePh').reshape((prodPhononSup,prodPhononSup,nModes,nBands,nBands)).swapaxes(3,4)
HePhWannier = cellWeightsEph[:,None,:,:,None] * cellWeightsEph[None,:,:,None,:] * HePhReduced[iReducedEph][:,iReducedEph]

# Calculate energies, eigenvectors and velocities for given k
def calcE(k):
    # Fourier transform to k:
    phase = np.exp((2j*np.pi)*np.dot(k,cellMap.T))
    H = np.tensordot(phase, Hwannier, axes=1)
    P = np.tensordot(phase, Pwannier,  axes=1)
    # Diagonalize and switch to eigen-basis:
    E,U = np.linalg.eigh(H)  # Diagonalize
    v = np.imag(np.einsum(
        'kba, kibc, kca -> kai', U.conj(), P, U, optimize="optimal"))  # diagonal only
    return E, U, v

# Calculate phonon energies and eigenvectors for given q
def calcPh(q):
    phase = np.exp((2j*np.pi)*np.tensordot(q,cellMapPh.T, axes=1))
    omegaSq, U = np.linalg.eigh(np.tensordot(phase, omegaSqR, axes=1))
    omegaPh = np.sqrt(np.maximum(omegaSq, 0.))
    return omegaPh, U

# Calculate e-ph matrix elements, along with ph and e energies, and e velocities
def realToFourierJDFTx(k1, k2):
    # Electrons:
    E1, U1, v1 = calcE(k1)
    E2, U2, v2 = calcE(k2)

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(k1[:,None,:] - k2[None,:,:])

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j*np.pi)*np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j*np.pi)*np.dot(k2,cellMapEph.T))
    normFac = np.sqrt(0.5/np.maximum(omegaPh,1e-6))
    normFac[:] = 1.

    g = np.einsum(
        'kKy, kac, Kbd, kKxy, kr, KR, rRxab -> kKycd',
        normFac, U1.conj(), U2, Uph, phase1.conj(), phase2, HePhWannier,
        optimize='optimal'
    )
    return g, omegaPh, E1, E2, v1, v2

def fourierToRealPhoebe(gkk_JDFTx,k1,k2):

    # set up the eigenvectors for rotation 
    E1, U1, v1 = calcE(k1)
    E2, U2, v2 = calcE(k2)

    # q = k1 - k2 -> k1 = k2 + q? This is sort of a negative q relationship
    q3 = k1[:,None,:] - k2[None,:,:]

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(q3)

    numKpts = k1.shape[0]

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((-2j * np.pi) * np.dot(k1, cellMapRe)).T/numKpts
    phase2 = np.exp((-2j * np.pi) * np.dot(q3, cellMapRp)).T/(numKpts*numKpts) # numKpts^2 = numQpts here

    UphInv = np.zeros(Uph.shape,dtype=np.complex128)
    for k1idx,kpt1 in enumerate(k1):
        for k2idx,kpt2 in enumerate(k2):
            UphInv[k1idx,k2idx] = np.linalg.inv(Uph[k1idx,k2idx])

    # here I think we need to flip which U2 or U1 is conj from the Phoebe notation on the website
    # as in footnote 16 of the wannier review article by marzari
    gRR_phoebe = np.einsum('kac, Kbd, kKxy, rk, RkK, kKycd -> rRxab',
        U1.conj(), U2, UphInv, phase1, phase2, gkk_JDFTx,
        optimize='optimal'
    )

#    Re = 300 
#    Rp = 10
#    x = 2 # atom index
#    w1 = 4 # wannier centers
#    w2 = 6
#    numBands = 8
#    gtemp = 0; 

#    for k1idx,kpt1 in enumerate(k1):
#        for k2idx,kpt2 in enumerate(k2):
#            for alpha in range(3): 
#                for nb1 in range(numBands):
#                    for nb2 in range(numBands):
#                        gtemp += phase1[Re,k1idx] * phase2[Rp,k1idx,k2idx] * U2[k2idx,w2,nb2] * U1[k1idx,w1,nb1].conj() * UphInv[k1idx,k2idx,x,alpha] * gkk_JDFTx[k1idx,k2idx,alpha,nb1,nb2]

    #print("gtemp ", gtemp, " vs einsum ", gRR_phoebe[Re,Rp,x,w1,w2], gtemp.real/gRR_phoebe[Re,Rp,x,w1,w2].real)

    return gRR_phoebe

def realToFourierPhoebe(gRR_phoebe,k1,k2):

    # set up the eigenvectors for rotation 
    E1, U1, v1 = calcE(k1)
    E2, U2, v2 = calcE(k2)

    # q = k1 - k2 -> k1 = k2 + q? This is sort of a negative q relationship
    q3 = k1[:,None,:] - k2[None,:,:]

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(q3)

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j * np.pi) * np.dot(k1, cellMapRe))
    phase2 = np.exp((2j * np.pi) * np.dot(q3, cellMapRp))
    phase1 = np.multiply(phase1, 1./cellWeightsRe.flatten())
    phase2 = np.multiply(phase2, 1./cellWeightsRp.flatten())

    # here I think we need to flip which U2 or U1 is conj from the Phoebe notation on the website
    # as in footnote 16 of the wannier review article by marzari 
    gkk_phoebe = np.einsum(
        'kac, Kbd, kKxy, kr, kKR, rRxab -> kKycd',
        U1, U2.conj(), Uph, phase1, phase2, gRR_phoebe,
        optimize='optimal'
    )

#    ik1 = 3
#    ik2 = 6
#    b1 = 2
#    b2 = 6
#    alpha = 2
#    gtemp = 0
#    numBands = 8

#    for R1idx,R1 in enumerate(cellMapRe.T):
#        for R2idx,R2 in enumerate(cellMapRp.T):
#            for x in range(3):
#                for nw1 in range(numBands):
#                    for nw2 in range(numBands):
#                        gtemp += phase1[ik1, R1idx] * phase2[ik1,ik2,R2idx] * U2[ik2,nw2,b2].conj() * U1[ik1,nw1,b1] * Uph[ik1,ik2,x,alpha] * gRR_phoebe[R1idx,R2idx,x,nw1,nw2]

#    print("gtemp ", gtemp, " vs einsum ", gkk_phoebe[ik1,ik2,alpha,b1,b2], gtemp.real/gkk_phoebe[ik1,ik2,alpha,b1,b2].real)

    return gkk_phoebe


# read in the k and q points from the original DFT calculation 
# read in the k points 
#qPoints = np.loadtxt("totalE.phononKpts")
kPoints = np.loadtxt("totalE.kPts",usecols=[2,3,4])

# interpolate jdftx to fourier space 
gkk_jdftx, omegaPh, E1, E2, v1, v2 = realToFourierJDFTx(kPoints, kPoints)
# transform to phoebe R basis 
gRR_phoebe = fourierToRealPhoebe(gkk_jdftx,kPoints,kPoints)
# transform using phoebe convention
gkk_phoebe = realToFourierPhoebe(gRR_phoebe,kPoints,kPoints)

print(np.absolute(gkk_jdftx[3,3,2,3,:]))
print(np.absolute(gkk_phoebe[3,3,2,3,:]))
print(np.absolute(gkk_jdftx[3,3,2,3,:])/np.absolute(gkk_phoebe[3,3,2,3,:]))
