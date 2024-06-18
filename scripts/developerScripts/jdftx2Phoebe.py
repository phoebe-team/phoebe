import numpy as np
import h5py
import sys

# Phoebe requires all these compoenents, which we can read in and reformat from JDFTx
#elBravaisVectors         Dataset {3, 1957}
#elDegeneracies           Dataset {1957, 1}
#kMesh                    Dataset {3, 1}
#numElBands               Dataset {SCALAR}
#numElectrons             Dataset {SCALAR}
#numSpin                  Dataset {SCALAR}

#fileFormat               Dataset {SCALAR}
#gWannier                 Dataset {1, 1470662016}

#numPhModes               Dataset {SCALAR}
#phBravaisVectors         Dataset {3, 1957}
#phDegeneracies           Dataset {1957, 1}
#qMesh                    Dataset {3, 1}

# needed for mass lookup -- mass is second item in each list
periodicTable =  {"H": [1, 1.008, 0.0002789062238058872], "He": [2, 4.002602, 1.215076833998857e-07], "Li": [3, 6.94, 0.001496682231674627], "Be": [4, 9.0121831, 0.0], "B": [5, 10.81, 0.0013805405232090412], "C": [6, 12.011, 8.000097592094282e-05], "N": [7, 14.007, 2.1290646750606633e-05], "O": [8, 15.999, 3.6319747512798014e-05], "F": [9, 18.998403163, 0.0], "Ne": [10, 20.1797, 0.0008278312013487067], "Na": [11, 22.98976928, 0.0], "Mg": [12, 24.305, 0.0007435878422415102], "Al": [13, 26.9815385, 0.0], "Si": [14, 28.085, 0.00020164270215294942], "P": [15, 30.973761998, 0.0], "S": [16, 32.06, 0.00018482784294313574], "Cl": [17, 35.45, 0.000619333162756516], "Ar": [18, 39.948, 3.4803742792164614e-05], "K": [19, 39.0983, 0.00016400010165430582], "Ca": [20, 40.078, 0.0002975638195255614], "Sc": [21, 44.955908, 0.0], "Ti": [22, 47.867, 0.00028645554052913216], "V": [23, 50.9415, 9.54832036799122e-07], "Cr": [24, 51.9961, 0.00013287039947669415], "Mn": [25, 54.938044, 0.0], "Fe": [26, 55.845, 8.24441020198494e-05], "Co": [27, 58.933194, 0.0], "Ni": [28, 58.6934, 0.00043070969156679167], "Cu": [29, 63.546, 0.00021093313034487347], "Zn": [30, 65.38, 0.0005931534544256652], "Ga": [31, 69.723, 0.00019712729693874742], "Ge": [32, 72.63, 0.0005867117795429008], "As": [33, 74.921595, 0.0], "Se": [34, 78.971, 0.00046053781437921013], "Br": [35, 79.904, 0.00016577469277381828], "Kr": [36, 83.798, 0.00024880741715297897], "Rb": [37, 85.4678, 0.00010969638953101122], "Sr": [38, 87.62, 6.099700710365833e-05], "Y": [39, 88.90584, 0.0], "Zr": [40, 91.224, 0.00034262838409586854], "Nb": [41, 92.90637, 0.0], "Mo": [42, 95.95, 0.0005968578764839594], "Tc": [43, 97.90721, 0.0], "Ru": [44, 101.07, 0.00040666639101881745], "Rh": [45, 102.9055, 0.0], "Pd": [46, 106.42, 0.0003094784331573286], "Ag": [47, 107.8682, 8.579859015398756e-05], "Cd": [48, 112.414, 0.00027153582954249383], "In": [49, 114.818, 1.2430878582037971e-05], "Sn": [50, 118.71, 0.00033408521010159045], "Sb": [51, 121.76, 6.607546197957849e-05], "Te": [52, 127.6, 0.0002839333322460794], "I": [53, 126.90447, 0.0], "Xe": [54, 131.293, 0.0002675478178911911], "Cs": [55, 132.90545196, 0.0], "Ba": [56, 137.327, 6.247983393130228e-05], "La": [57, 138.90547, 4.5917359997528195e-08], "Ce": [58, 140.116, 2.2504655383416496e-05], "Pr": [59, 140.90766, 0.0], "Nd": [60, 144.242, 0.00023237352095226282], "Pm": [61, 144.91276, 0.0], "Sm": [62, 150.36, 0.0003348898354048051], "Eu": [63, 151.964, 4.327943128302847e-05], "Gd": [64, 157.25, 0.00012767481945965138], "Tb": [65, 158.92535, 0.0], "Dy": [66, 162.5, 5.1980645724531205e-05], "Ho": [67, 164.93033, 0.0], "Er": [68, 167.259, 7.232484216696818e-05], "Tm": [69, 168.93422, 0.0], "Yb": [70, 173.045, 8.560802725546705e-05], "Lu": [71, 174.9668, 8.300719193988442e-07], "Hf": [72, 178.49, 5.253848527005391e-05], "Ta": [73, 180.94788, 3.671578002866865e-09], "W": [74, 183.84, 6.96679689149215e-05], "Re": [75, 186.207, 2.708496655430222e-05], "Os": [76, 190.23, 7.452361349988645e-05], "Ir": [77, 192.217, 2.537864205006749e-05], "Pt": [78, 195.084, 3.428660338142578e-05], "Au": [79, 196.966569, 0.0], "Hg": [80, 200.592, 6.52279806399407e-05], "Tl": [81, 204.38, 2.2233484775588543e-05], "Pb": [82, 207.2, 1.9437786146643996e-05], "Bi": [83, 208.9804, 0.0], "Po": [84, 209.0, 0.0], "At": [85, 210.0, 0.0], "Rn": [86, 222.0, 0.0], "Fr": [87, 223.0, 0.0], "Ra": [88, 226.0, 0.0], "Ac": [89, 227.0, 0.0], "Th": [90, 232.0377, 1.492878990056669e-08], "Pa": [91, 231.03588, 0.0], "U": [92, 238.02891, 1.1564605377712275e-06], "Np": [93, 237.0, 0.0], "Pu": [94, 244.0, 0.0], "Am": [95, 243.0, 0.0], "Cm": [96, 247.0, 0.0], "Bk": [97, 247.0, 0.0], "Cf": [98, 251.0, 0.0], "Es": [99, 252.0, 0.0], "Fm": [100, 257.0, 0.0], "Md": [101, 258.0, 0.0], "No": [102, 259.0, 0.0], "Lr": [103, 262.0, 0.0], "Rf": [104, 267.0, 0.0], "Db": [105, 268.0, 0.0], "Sg": [106, 271.0, 0.0], "Bh": [107, 274.0, 0.0], "Hs": [108, 269.0, 0.0], "Mt": [109, 276.0, 0.0]}


hf = h5py.File('jdftx.elph.phoebe.hdf5', 'w')
phaseConvention = 1
hf.create_dataset('phaseConvention',data=phaseConvention)

R = np.zeros((3,3))
iLine = 0
refLine = -10
Rdone = False
collectIons = False
masses = []

# NOTE: YOU MUST MANUALLY CHANGE THIS VALUE TO THE NUMBER OF EXCLUDED BANDS!
excludeBands = 16

# Read kpoints information and chemical potential, nelec, spin, etc from JDFTx
for line in open('totalE.out'):
    if line.startswith('\tFillingsUpdate:'):
        mu = float(line.split()[2])
    if line.startswith('nElectrons:'):
        nElec = int(line.split()[1].split('.')[0])
    if line.startswith('spintype'):
        if(line.split()[1] == 'no-spin'): nSpin = 1
        else: nSpin = 2
    if line.startswith("ion ") and collectIons:
        ionName = line.split()[1]
        masses.append(periodicTable[ionName][1])
    if line.startswith('forces-output-coords'):
        collectIons = True
    if line.startswith('kpoint-folding'):
        collectIons = False
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
    if line.find('Initializing the Grid') >= 0:
            refLine = iLine
    if not Rdone:
            rowNum = iLine - (refLine+2)
            if rowNum>=0 and rowNum<3:
                    R[rowNum,:] = [ float(x) for x in line.split()[1:-1] ]
            if rowNum==3:
                    Rdone = True
    iLine += 1

kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

# Read the MLWF cell map, weights and Hamiltonian ------------------
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(int)

Wwannier = np.fromfile("wannier.mlwfCellWeights")
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)
Hreduced = np.fromfile("wannier.mlwfH").reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]
Hwannier = Hwannier.astype(complex)

# save the Wannier hamiltonian + R vectors and weights to HDF5 for Phoebe
# Note: cell weights are one because they were applied above when Hwannier was expanded

# Phoebe expects these in cartesian coordinates (Bohr)
# Additionally, JDFTx uses a phase convention which expects a H(0,R) while phoebe expects
# H(R,0) -- we account for this by adding a -1 to the phase below.
cellMap = np.einsum("xy,Rx->Ry",-1*R.T,cellMap)

hf.create_dataset('elBravaisVectors', data=cellMap.T)
hf.create_dataset('elDegeneracies', data=np.ones(nCells).reshape(nCells,1))
hf.create_dataset('kMesh', data=kfold.reshape(3,1))
hf.create_dataset('numElBands', data=nBands)
hf.create_dataset('numSpin', data=nSpin)
hf.create_dataset('numElectrons', data=nElec - excludeBands)
# in addition to regular Phoebe quantities, we will also read this from HDF5
hf.create_dataset('chemicalPotential', data=mu*2.)  # convert to Ry
hf.create_dataset('wannierHamiltonian', data=Hwannier*2.) # convert to Ry

print(nElec)
print(mu*27.21)

# Read the elph cell map, weights and matrix elements ------------------
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

# to make use of the generic function in Phoebe for reading elph matrix elements,
# we also write some phonon information to the file
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[0,1,2]).astype(int)
nCellsPh = cellMapPh.shape[0]
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))

# --- Read e-ph cell weights:
nAtoms = nModes // 3
cellWeightsEph = np.fromfile("wannier.mlwfCellWeightsPh").reshape((nCellsEph,nBands,nAtoms)).swapaxes(1,2)
cellWeightsEph = np.repeat(cellWeightsEph.reshape((nCellsEph,nAtoms,1,nBands)), 3, axis=2)  # repeat atom weights for 3 directions
cellWeightsEph = cellWeightsEph.reshape((nCellsEph,nModes,nBands))  # combine nAtoms x 3 into single dimension: nModes
iReducedEph = np.dot(np.mod(cellMapEph, phononSup[None,:]), phononSupStride)
HePhReduced = np.fromfile('wannier.mlwfHePh').reshape((prodPhononSup,prodPhononSup,nModes,nBands,nBands)).swapaxes(3,4)
HePhWannier = cellWeightsEph[:,None,:,:,None] * cellWeightsEph[None,:,:,None,:] * HePhReduced[iReducedEph][:,iReducedEph]

# reload this in cartesian coords to write it easily
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[0,1,2]).astype(int)
# convert the cell map to cartesian coordinates
cellMapEph = np.einsum("xy,Rx->Ry",R.T,cellMapEph)

# write the elph information
hf.create_dataset('elphDegeneracies', data=np.ones(nCellsEph).reshape(nCellsEph,1))
hf.create_dataset('elphBravaisVectors', data=cellMapEph.reshape(3,-1))  # reshaping this way thrwarts a problem with Eigen and row/col major
hf.create_dataset("fileFormat", data=1)  # this tells Phoebe to read in all the data at once rather than in chunks

# transpose along the nAtoms,3 block to account for phoebe's expectation that the dynamical matrix will be
# in the order nAtom, nAtom, 3, 3 rather than nAtom, 3, nAtom, 3 -- this is due to a row/col major ordering dilemma
# basically, we need this block when flattened to have the ordering 1x 2x 3x 1y 2y 3y 1z 2z 3z
# instead of the common 1x 1y 1 z 2x 2y 2z...
HePhWannier = HePhWannier.reshape(HePhWannier.shape[0],HePhWannier.shape[1],nAtoms,3,HePhWannier.shape[3],HePhWannier.shape[4])

# now apply the species masses
massAmuToRy = 1.660538782e-27 / 9.10938215e-31 / 2.
HePhWannier = np.einsum('RraxWw,a->RraxWw',HePhWannier,np.sqrt(massAmuToRy * np.array(masses)))

# write to file
HePhWannier = HePhWannier.flatten(order='C')*2.  # convert Ha->Ry
HePhWannier = HePhWannier.astype(complex)
hf.create_dataset('gWannier', data=HePhWannier.reshape(1,HePhWannier.shape[0]))

# write some phonon related information
# convert the cell map to cartesian coordinates
cellMapPh = np.einsum("xy,Rx->Ry",-1*R.T,cellMapPh)

hf.create_dataset('phBravaisVectors', data=cellMapPh.T)
hf.create_dataset('phDegeneracies', data=np.ones(nCellsPh).reshape(nCellsPh,1))
hf.create_dataset('qMesh', data=phononSup.reshape(3,1))
hf.create_dataset('numPhModes', data=nModes)
# these contain the atomic species masses, therefore we need to convert an extra factor of 2
# we also need to reorder them to match phoebe's expectation
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)
omegaSqR = omegaSqR.reshape((nCellsPh,nAtoms,3,nAtoms,3))
hf.create_dataset('forceConstants', data=omegaSqR*2.*2.) # convert to Ry

hf.close()

