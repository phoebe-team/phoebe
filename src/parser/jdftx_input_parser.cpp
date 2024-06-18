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

  // load in the data written to jdftx.elph.phoebe.hdf5 by the conversion script
  // ========================================================================

  // set up containers to read data into 
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> FC2s; // nCells, nAtoms, 3, nAtoms, 3
  std::vector<std::vector<double>> cellMap;
  Eigen::Vector3i qMesh; 
  int nCells = 0;
  int nModes = 0; 

  #ifdef HDF5_AVAIL

  std::string fileName = context.getPhFC2FileName();
  if (fileName.empty()) {
    Error("Check your path, jdftx.elph.phoebe.hdf5 not found at " + fileName);
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  try {

    // Open the hdf5 file
    HighFive::File file(fileName, HighFive::File::ReadOnly);

    // Set up hdf5 datasets
    HighFive::DataSet dforceContants = file.getDataSet("/forceConstants");
    HighFive::DataSet dCellMap = file.getDataSet("/phBravaisVectors");
    HighFive::DataSet dqMesh = file.getDataSet("/qMesh");
    HighFive::DataSet dnModes = file.getDataSet("/numPhModes");

    // read in the data 
    dforceContants.read(FC2s);
    dCellMap.read(cellMap);
    dqMesh.read(qMesh);
    dnModes.read(nModes);

    nCells = FC2s.size(); 

    // Sanity check data that has been read in 
    if(int(cellMap[0].size()) != nCells) {
      Error("Somehow, the number of R vectors in your JDFTx ph cell map does\n"
          "not match your force constant file dimensions! Check the input JDFTx files.");
    }
    if(cellMap.size() != 3) { Error("JDFTx ph cellMap does not have correct first dimension."); }

  } catch (std::exception &error) {
    if(mpi->mpiHead()) std::cout << error.what() << std::endl;
    Error("Issue found while reading jdftx.elph.phoebe.hdf5. Make sure it exists at " + fileName +
          "\n and is not open by some other persisting processes.");
  }

  #else
    Error("To use JDFTx input for phonon force constants, you must build the code with HDF5.");
  #endif

  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed harmonic phonon JDFTx files.\n" << std::endl;
  }

  // reformat to eigen containers needed by phononH0
  Eigen::MatrixXd cellMapReformat(3,nCells);
  Eigen::Tensor<double,5> forceConstants(3, 3, numAtoms, numAtoms, nCells);

  // multiply the force constants by the species masses 
  for(int iR = 0; iR<nCells; iR++) {
    for(int ia = 0; ia<numAtoms; ia++) {
      for(int ja = 0; ja<numAtoms; ja++) {
        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {

            // the JDFTx force constants also already include the atomic masses, 
            // and are in fact C/sqrt(M_i * M_j)
            int iSpecies = atomicSpecies(ia);
            int jSpecies = atomicSpecies(ja);
            forceConstants(i,j,ia,ja,iR) = FC2s[iR][ia][i][ja][j] * sqrt(speciesMasses(iSpecies) * speciesMasses(jSpecies));  
          }
        }
      }
    } 
    cellMapReformat(0,iR) = cellMap[0][iR];
    cellMapReformat(1,iR) = cellMap[1][iR];
    cellMapReformat(2,iR) = cellMap[2][iR];
  }

  Eigen::VectorXd cellWeights(nCells);
  cellWeights.setConstant(1.);

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, 
                           forceConstants, qMesh,
                           cellMapReformat, cellWeights);

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

  // parse the crystal structure from totalE.out 
  // ========================================================================
  Crystal crystal = parseCrystal(context);
  auto directUnitCell = crystal.getDirectUnitCell();
  auto atomicPositions = crystal.getAtomicPositions();

  // load in the data written to jdftx.elph.phoebe.hdf5 by the conversion script
  // ========================================================================

  // set up containers to read data into 
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

  // the path to wannier jdftx file 
  std::string fileName = context.getElectronH0Name();
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
  if(spinFactor == 2) spinOrbit = true; 
  context.setHasSpinOrbit(spinOrbit);
  //if (!spinOrbit) { // the case of spin orbit
  //  nElectrons /= 2;  // apparently this is unneeded and problematic
  //}
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

  return std::make_tuple(crystal, electronH0);
}

/* Helper function to read crystal class information  */
Crystal JDFTxParser::parseCrystal(Context& context) {

  // TODO we need to fix this before publishing 
  std::string fileName = context.getJDFTxScfOutFile();

  // open input file
  std::ifstream infile(fileName);

  if (fileName.empty() || !infile.is_open()) {
    Error("totalE.out file not found at " + fileName);
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

  // convert unit cell positions to cartesian, in bohr
  for (int i = 0; i < numAtoms; i++) {
    // copy into Eigen structure
    atomicSpecies(i) = tempSpecies[i];
    // convert to cartesian
    Eigen::Vector3d temp(tempPositions[i][0], tempPositions[i][1], tempPositions[i][2]);
    Eigen::Vector3d temp2 = directUnitCell * temp; // lattice already in Bohr
    atomicPositions(i, 0) = temp2(0);
    atomicPositions(i, 1) = temp2(1);
    atomicPositions(i, 2) = temp2(2);
  }

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