#include <algorithm> // to use .remove_if
#include <cmath>     // round()
#include <cstdlib>   // abs()
#include <fstream>
#include <string>
#include <vector>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "particle.h"
#include "periodic_table.h"
#include "points.h"
#include "jdftx_input_parser.h"
#include "utilities.h"

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

std::tuple<Crystal, PhononH0> JDFTxParser::parsePhHarmonic(Context &context) {
  //  Here we read the dynamical matrix of inter-atomic force constants
  //	in real space.

  // TODO we need to fix this before publishing 
  std::string directory = context.getJDFTxDirectory();

  // parse the crystal structure from totalE.out 
  // ========================================================================
  Crystal crystal = parseCrystal(context);
  int numAtoms = crystal.getNumAtoms();
  auto directUnitCell = crystal.getDirectUnitCell();
  auto atomicSpecies = crystal.getAtomicSpecies();
  auto speciesMasses = crystal.getSpeciesMasses();

  // Here there is no born charge data, so we set this to zero 
  // TODO This would be saved in totalE.Zeff and totalE.epsInf (the dielectric tensor)
  Eigen::Matrix3d dielectricMatrix;
  dielectricMatrix.setZero();

  // read in phonon supercell from phonon.out
  // ========================================================================

  std::string fileName = directory + "phonon.out";

  // open input file
  std::ifstream infile(fileName);

  if (fileName.empty() || !infile.is_open()) {
    Error("phonon.out file not found in " + directory);
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  std::string line;
  std::vector<std::string> lineSplit;

  Eigen::Vector3i qCoarseGrid; 

  while (std::getline(infile, line)) {

    // get the phonon supercell size
    if(line.find("supercell") != std::string::npos && line.find("\\") != std::string::npos) {
      auto mesh = split(line, ' ');
      qCoarseGrid[0] = std::stoi(mesh[1]);
      qCoarseGrid[1] = std::stoi(mesh[2]);
      qCoarseGrid[2] = std::stoi(mesh[3]);
    }
  }
  infile.close();
  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error("Found a invalid coarse q grid: " + std::to_string(qCoarseGrid(0)) + " " 
          + std::to_string(qCoarseGrid(1)) + " " + std::to_string(qCoarseGrid(2)));
  }

  // Read in the cellMap/R vector file
  // =============================================================

  fileName = directory + "totalE.phononCellMap";

    // open the file
  infile.open(fileName);
  if (fileName.empty() || !infile.is_open()) {
    Error("*.phononCellMap file not found in " + directory);
  }

  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  int nCells = 0;
  std::vector<std::vector<int>> cellMap;
  int temp1, temp2, temp3, cellMap1, cellMap2, cellMap3;

  // read in the three numbers for each cell
  std::getline(infile,line); // there's one header line
  while( !infile.eof() ) {
    std::getline(infile, line);
    if(infile.eof()) break;
    std::istringstream iss(line);
    iss >> cellMap1 >> cellMap2 >> cellMap3 >> temp1 >> temp2 >> temp3;
    std::vector<int> temp{cellMap1,cellMap2,cellMap3};
    cellMap.push_back(temp);
    nCells++;
  }
  infile.close();

  // read in dynamical matrix file
  // ========================================================================

  // check that the file exists 
  fileName = directory + "totalE.phononOmegaSq";
  // open input file
  infile.open(fileName, std::ios::in | std::ios::binary);

  if (fileName.empty() || !infile.is_open()) {
    Error("*.phononOmegaSq not found at " + fileName);
  }

  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  // the size of the totalE.phononOmegaSq file is
  // ncells * nModes * nModes, which we know.
  size_t size = nCells * 3 * 3 * numAtoms * numAtoms;
  std::vector<double> buffer(size);
  if(!infile.read((char*)buffer.data(), size*sizeof(double))) {
    Error("Problem found when reading in phononOmegaSq file!");
  }
  infile.close();

  // reshape the force constants to match the expected phoebe format 
  // first we read it in as JDFTx expects it, then below, we swap the dimension
  // arguments and the numAtoms arguments for use in PhononH0
  // Phoebe uses numAtoms, numAtoms, 3, 3 
  auto mapped_t = Eigen::TensorMap<Eigen::Tensor<double,5>>(&buffer[0],
                                          numAtoms, 3, numAtoms, 3, nCells);

  // force constants matrix 
  Eigen::Tensor<double,5> forceConstantsInitial = mapped_t;
  Eigen::array<int, 5> shuffling({1, 3, 0, 2, 4}); 
  Eigen::Tensor<double,5> forceConstants = forceConstantsInitial.shuffle(shuffling);

  // JDFTx uses hartree, so here we must convert to Ry
  // Additionally, there is a second factor of 2 which accounts for the 
  // factor of 2 difference in mass between Rydberg and Hartree atomic units
  // the JDFTx force constants alreadyx include the atomic masses
  forceConstants = forceConstants * 2. * 2.; 

  Eigen::MatrixXd bravaisVectors(3,nCells); 
  // convert cell map to cartesian coordinates 
  for(int iR = 0; iR<nCells; iR++) {

    // use cellMap to get R vector indices
    Eigen::Vector3i Rcrys = {cellMap[iR][0],cellMap[iR][1],cellMap[iR][2]};

    // convert from crystal coordinate indices to cartesian coordinates
    // to match Phoebe conventions 
    Eigen::Vector3d Rcart = Rcrys[0] * directUnitCell.col(0) + Rcrys[1] * directUnitCell.col(1) +  Rcrys[2] * directUnitCell.col(2);
    bravaisVectors(0,iR) = -Rcart(0); 
    bravaisVectors(1,iR) = -Rcart(1); 
    bravaisVectors(2,iR) = -Rcart(2); 


    for(int ia = 0; ia<numAtoms; ia++) {
      for(int ja = 0; ja<numAtoms; ja++) {
        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {

            // the JDFTx force constants also already include the atomic masses, 
            // and are in fact C/sqrt(M_i * M_j)
            int iSpecies = atomicSpecies(ia);
            int jSpecies = atomicSpecies(ja);
            forceConstants(i,j,ia,ja,iR) = forceConstants(i,j,ia,ja,iR) * sqrt(speciesMasses(iSpecies) * speciesMasses(jSpecies));
          }
        }
      }
    } 
  }

  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed harmonic phonon JDFTx files.\n" << std::endl;
  }

  Eigen::VectorXd cellWeights(nCells);
  cellWeights.setConstant(1.);

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, 
                           forceConstants, qCoarseGrid,
                           bravaisVectors, cellWeights);

  // no need to apply a sum rule, as the JDFTx matrix elements have already
  // had one applied internally
  if(context.getSumRuleFC2() != "none") {
    Warning("The phonon force constants from JDFTx already have a sum rule applied.\n"
      "Therefore, the sum rule chosen from input will be ignored.");
  }
  if(speciesMasses.size() > 1) { 
    Warning("You have a material which has more than one species, and therefore may be polar.\n"
    "Currently, we are not set up to parse the effective charges and dielectric matrix from JDFTx --\n"
    "however this should be possible, so let us know if you need this capability.");
  }
  return {crystal, dynamicalMatrix};
}

std::tuple<Crystal, ElectronH0Wannier> JDFTxParser::parseElHarmonicWannier(
                                                      Context &context, 
                                                      [[maybe_unused]] Crystal *inCrystal) {

  // the directory where jdftx inputs live 
  std::string directory = context.getJDFTxDirectory();

  // parse the crystal structure from totalE.out 
  // ========================================================================
  Crystal crystal = parseCrystal(context);
  auto directUnitCell = crystal.getDirectUnitCell();
  auto atomicPositions = crystal.getAtomicPositions();

  // load in the data written to jdftx.elph.phoebe.hdf5 by the conversion script
  // ========================================================================

  // set up containers to read data into 

  // here, we do something weird -- if SOC is used, JDFTx's wannier files become 
  // complex rather than real. Depending on spinFac, we change how this is read in 
  // We read these into vectors because if read into Eigen, things become flipped 
  std::vector<std::vector<std::vector<std::complex<double>>>> HWannier;
  std::vector<std::vector<double>> cellMap;
  Eigen::Vector3i kMesh; 
  int nCells = 0;
  int nWannier = 0;
  int nBands = 0; 
  int nElectrons = 0; 
  int spinFactor = 2;
  double fermiLevel;

  #ifdef HDF5_AVAIL

  std::string fileName = directory + "jdftx.elph.phoebe.hdf5";
  if (fileName.empty()) {
    Error("Check your path, jdftx.elph.phoebe.hdf5 not found at " + fileName);
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  try {

    // Open the hdf5 file
    HighFive::File file(fileName, HighFive::File::ReadOnly);

    // Set up hdf5 datasets
    HighFive::DataSet dHwannier = file.getDataSet("/wannierHamiltonian");
    HighFive::DataSet dCellMap = file.getDataSet("/elBravaisVectors");
    HighFive::DataSet dkMesh = file.getDataSet("/kMesh");
    HighFive::DataSet dnElec = file.getDataSet("/numElectrons");
    HighFive::DataSet dnBands = file.getDataSet("/numElBands");
    HighFive::DataSet dnSpin = file.getDataSet("/numSpin");
    HighFive::DataSet dmu = file.getDataSet("/chemicalPotential");

    // read in the data 
    dHwannier.read(HWannier);
    dCellMap.read(cellMap);
    dkMesh.read(kMesh);
    dnElec.read(nElectrons);
    dnBands.read(nBands);
    dnSpin.read(spinFactor);
    dmu.read(fermiLevel);

    nCells = HWannier.size(); 
    nWannier = HWannier[0].size(); 

    // Sanity check data that has been read in 
    if(int(cellMap[0].size()) != nCells) {
      Error("Somehow, the number of R vectors in your JDFTx elWannier cell map does\n"
          "not match your wannier.mlwfH file!");
    }
    if(cellMap.size() != 3) { Error("JDFTx el cellMap does not have correct first dimension."); }

  } catch (std::exception &error) {
    if(mpi->mpiHead()) std::cout << error.what() << std::endl;
    Error("Issue found while reading jdftx.elph.phoebe.hdf5. Make sure it exists at " + fileName +
          "\n and is not open by some other persisting processes.");
  }

  #else
    Error("To use JDFTx input for Wannier Hamiltonian, you must build the code with HDF5.");
  #endif

  // close out the function and return everything 
  // ========================================================================

  // For now, we do not have access to this information 
  // (the position matrix elements in the Wannier basis) from JDFTx. However, 
  // the function that uses this is never called, so we just send this as 
  // zeros to the constructor
  //Eigen::Tensor<std::complex<double>, 4> rMatrix(3, nCells, nWannier, nWannier);
  //rMatrix.setZero();

  bool spinOrbit = false; 
  if(spinFactor == 1) spinOrbit = true; 
  context.setHasSpinOrbit(spinOrbit);
  if (!spinOrbit) { // the case of spin orbit
    nElectrons /= 2;
  }
  context.setNumOccupiedStates(nElectrons);

  // if the user didn't set the Fermi level, we do it here.
  if (std::isnan(context.getFermiLevel()))
    context.setFermiLevel(fermiLevel);

  // copy the HWannier and cell Map into Eigen containers for the ElectronWannierH0 constructor
  Eigen::MatrixXd cellMapReformat(3,nCells);
  Eigen::Tensor<std::complex<double>, 3> h0R(nCells, nWannier, nWannier);

  for(int iCell = 0; iCell < nCells; iCell++) {
    for(int iw1 = 0; iw1 < nWannier; iw1++) {
      for(int iw2 = 0; iw2 < nWannier; iw2++) {
        h0R(iCell,iw1,iw2) = HWannier[iCell][iw1][iw2]; 
      }
    }
    cellMapReformat(0,iCell) = cellMap[0][iCell];
    cellMapReformat(1,iCell) = cellMap[1][iCell];
    cellMapReformat(2,iCell) = cellMap[2][iCell];
  }

  // for JDFTx, cell weights already are applied to HWannier, therefore, 
  // we set them to one 
  Eigen::VectorXd cellWeights(nCells); cellWeights.setConstant(1.);

  ElectronH0Wannier electronH0(directUnitCell, cellMapReformat, cellWeights, h0R);

  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed JDFTx electronic Hamiltonian files.\n" << std::endl;
  }

  Kokkos::Profiling::popRegion();
  return std::make_tuple(crystal, electronH0);
}

/* Helper function to read crystal class information  */
Crystal JDFTxParser::parseCrystal(Context& context) {

  // TODO we need to fix this before publishing 
  std::string directory = context.getJDFTxDirectory();
  std::string fileName = directory + "totalE.out";

  // open input file
  std::ifstream infile(fileName);

  if (fileName.empty() || !infile.is_open()) {
    Error("totalE.out file not found at " + directory);
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  std::string line;
  std::vector<std::string> lineSplit;

  // read in direct unit cell from totalE.out
  // ========================================================================
  Eigen::Matrix3d directUnitCell;

  // we don't know # of atoms ahead of time
  std::vector<std::vector<double>> tempPositions;
  std::vector<int> tempSpecies;
  std::vector<std::string> speciesNames;
  bool unitCellFound = false;

  while (std::getline(infile, line)) {
    // check for lattice vectors
    if (line.find("---------- Initializing the Grid ----------")
        != std::string::npos && !unitCellFound) {

      unitCellFound = true; // make sure we only read first instance
      // skip a line, then read in lattice vectors
      std::getline(infile,line);
      // read in lattice vectors
      for(int i = 0; i < 3; i++) {
        std::getline(infile,line);
        auto lv = split(line, ' ');
        for (int j = 0; j < 3; j++) {
          directUnitCell(i,j) = std::stod(lv[j+1]);
        }
      }
    }

    // check if the line contains ionic positions
    if (line.find("forces-output-coords") != std::string::npos) {
      // next lines contain ion positions in lattice coords
      // and atomic species
      while(std::getline(infile,line)) {

        // if ion is not in the line, we have all the lines now
        if(line.find("ion ") == std::string::npos)
          break;

        // otherwise, extract the positions
        auto pos = split(line, ' ');
        // if the unique species is not in this list, add it
        if (std::find(speciesNames.begin(), speciesNames.end(), pos[1]) ==
            speciesNames.end()) {
          speciesNames.push_back(pos[1]);
        }
        // add the atomic position and species type # for this
        // position to the list
        // save the atom number of this species
        tempSpecies.push_back(std::find(speciesNames.begin(),
              speciesNames.end(), pos[1]) - speciesNames.begin());

        // save position
        std::vector<double> temp;
        for(int j = 0; j < 3; j++) temp.push_back(std::stod(pos[j+2]));
        tempPositions.push_back(temp);
      }
    }
  }
  infile.close();

  // ready final data structures
  int numElements = speciesNames.size();
  int numAtoms = tempPositions.size();

  Eigen::VectorXd speciesMasses(numElements);
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);

  PeriodicTable periodicTable;
  for (int i = 0; i < numElements; i++)
    speciesMasses[i] = periodicTable.getMass(speciesNames[i]) * massAmuToRy;

  speciesMasses.setConstant(1.);

  directUnitCell.transposeInPlace();

  // convert unit cell positions to cartesian, in bohr
  for (int i = 0; i < numAtoms; i++) {
    // copy into Eigen structure
    atomicSpecies(i) = tempSpecies[i];
    // convert to cartesian
    Eigen::Vector3d temp(tempPositions[i][0], tempPositions[i][1],
                         tempPositions[i][2]);
    Eigen::Vector3d temp2 =
        directUnitCell.transpose() * temp; // lattice already in Bohr
    atomicPositions(i, 0) = temp2(0);
    atomicPositions(i, 1) = temp2(1);
    atomicPositions(i, 2) = temp2(2);
  }

  // unit cell of JDFTx is transposed wrt QE/VASP definition
  //directUnitCell.transposeInPlace();

  // Here there is no born charge data, so we set this to zero 
  // This would be saved in totalE.Zeff and totalE.epsInf (the dielectric tensor) 
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  // Now we do postprocessing
  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, bornCharges);
  crystal.print();

  return crystal; 

}