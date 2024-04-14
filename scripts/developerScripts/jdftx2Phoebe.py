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

hf = h5py.File('jdftx.elph.phoebe.hdf5', 'w')
phaseConvention = 1
hf.create_dataset('phaseConvention',data=phaseConvention)

# Read kpoints information and chemical potential, nelec, spin, etc from JDFTx
for line in open('totalE.out'):
    if line.startswith('\tFillingsUpdate:'):
        mu = float(line.split()[2])
    if line.startswith('nElectrons:'):
        nElec = int(line.split()[1].split('.')[0])
    if line.startswith('spintype'):
        if(line.split()[1] == 'no-spin'): nSpin = 2
        else: nSpin = 1
    if line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
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
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,3:6].astype(float)

hf.create_dataset('elBravaisVectors', data=cellMap.T)
hf.create_dataset('elDegeneracies', data=np.ones(nCells).reshape(nCells,1)) 
hf.create_dataset('kMesh', data=kfold.reshape(3,1)) 
hf.create_dataset('numElBands', data=nBands)  
hf.create_dataset('numSpin', data=nSpin)  
hf.create_dataset('numElectrons', data=nElec) 
# in addition to regular Phoebe quantities, we will also read this from HDF5
hf.create_dataset('chemicalPotential', data=mu*2.)  # convert to Ry
hf.create_dataset('wannierHamiltonian', data=Hwannier*2.) # convert to Ry

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
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)

# --- Read e-ph cell weights:
nAtoms = nModes // 3
cellWeightsEph = np.fromfile("wannier.mlwfCellWeightsPh").reshape((nCellsEph,nBands,nAtoms)).swapaxes(1,2)
cellWeightsEph = np.repeat(cellWeightsEph.reshape((nCellsEph,nAtoms,1,nBands)), 3, axis=2)  # repeat atom weights for 3 directions
cellWeightsEph = cellWeightsEph.reshape((nCellsEph,nModes,nBands))  # combine nAtoms x 3 into single dimension: nModes
iReducedEph = np.dot(np.mod(cellMapEph, phononSup[None,:]), phononSupStride)
HePhReduced = np.fromfile('wannier.mlwfHePh').reshape((prodPhononSup,prodPhononSup,nModes,nBands,nBands)).swapaxes(3,4)
HePhWannier = cellWeightsEph[:,None,:,:,None] * cellWeightsEph[None,:,:,None,:] * HePhReduced[iReducedEph][:,iReducedEph]

# reload this in cartesian coords to write it easily
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[3,4,5]).astype(float)

# write the elph information 
hf.create_dataset('elphDegeneracies', data=np.ones(nCellsEph).reshape(nCellsEph,1))
hf.create_dataset('elphBravaisVectors', data=cellMapEph.reshape(3,-1))  # reshaping this way thrwarts a problem with Eigen and row/col major
hf.create_dataset("fileFormat",data=1)  # this tells Phoebe to read in all the data at once rather than in chunks 
HePhWannier = HePhWannier.flatten()*2.  # convert Ha->Ry
HePhWannier = HePhWannier.astype(complex)
hf.create_dataset('gWannier', data=HePhWannier.reshape(1,HePhWannier.shape[0]))

# write some phonon related information 
# reload this in cartesian coords to write it easily
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[3,4,5]).astype(float)

hf.create_dataset('phBravaisVectors', data=cellMapPh.T)
hf.create_dataset('phDegeneracies', data=np.ones(nCellsPh).reshape(nCellsPh,1))
hf.create_dataset('qMesh', data=phononSup.reshape(3,1))
hf.create_dataset('numPhModes', data=nModes)

hf.close()


