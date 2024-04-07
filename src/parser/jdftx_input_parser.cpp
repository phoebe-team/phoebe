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

std::tuple<Crystal, PhononH0> JDFTxParser::parsePhHarmonic(Context &context) {
  //  Here we read the dynamical matrix of inter-atomic force constants
  //	in real space.

  // TODO we need to fix this before publishing 
  std::string directory = "/mnt/ceph/users/jcoulter/4.Cu/9.jdftx/";

  std::string fileName = directory + "phonon.out";
  if (fileName.empty())
    Error("phonon.out file not found in " + directory);

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open())
    Error("totalE.out file not found");
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  // read in direct unit cell from totalE.out
  // ========================================================================
  Eigen::Matrix3d directUnitCell;

  // we don't know # of atoms ahead of time
  std::vector<std::vector<double>> tempPositions;
  std::vector<int> tempSpecies;
  std::vector<std::string> speciesNames;

  bool unitCellFound = false;
  Eigen::Vector3i qCoarseGrid; qCoarseGrid.setZero();

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
    // get the phonon supercell size
    if(line.find("supercell") != std::string::npos && line.find("\\") != std::string::npos) {

      auto mesh = split(line, ' ');
      std::cout << line << std::endl;
      qCoarseGrid[0] = std::stoi(mesh[1]);
      qCoarseGrid[1] = std::stoi(mesh[2]);
      qCoarseGrid[2] = std::stoi(mesh[3]);
    }

    // check if the line contains ionic positions
    if (line.find("initial-state ") != std::string::npos) {
      // next lines contain ion positions in lattice coords
      // and atomic species
      while(std::getline(infile,line)) {
        // if ion is not in the line, we have all the lines now
        std::cout << line << std::endl;

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
  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error("Found a invalid coarse q grid: " + std::to_string(qCoarseGrid(0)) + " " 
          + std::to_string(qCoarseGrid(1)) + " " + std::to_string(qCoarseGrid(2)));
  }

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
    Eigen::Vector3d temp(tempPositions[i][0], tempPositions[i][1],
                         tempPositions[i][2]);
    Eigen::Vector3d temp2 =
        directUnitCell.transpose() * temp; // lattice already in Bohr
    atomicPositions(i, 0) = temp2(0);
    atomicPositions(i, 1) = temp2(1);
    atomicPositions(i, 2) = temp2(2);
  }

  // Read if hasDielectric -- for now, let's just assume no. However, 
  // there should be some way to supply this information. 
  Eigen::Matrix3d dielectricMatrix;
  dielectricMatrix.setZero();
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  // Read in the cellMap/R vector file
  // =============================================================
  fileName = directory + "totalE.phononCellMap";
  if (fileName.empty()) {
    Error("Must provide a cellMap file name");
  }

  // open the file
  infile.open(fileName);
  if(!infile.is_open()) {
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
  // arguments and teh numAtoms arguments for use in PhononH0
  auto mapped_t = Eigen::TensorMap<Eigen::Tensor<double,5>>(&buffer[0],
                                          numAtoms, 3, numAtoms, 3, nCells);

  // force constants matrix 
  Eigen::Tensor<double,5> forceConstantsInitial = mapped_t;
  Eigen::array<int, 5> shuffling({1, 3, 0, 2, 4}); 
  Eigen::Tensor<double,5> forceConstants = forceConstantsInitial.shuffle(shuffling);

  // JDFTx uses hartree, so here we must convert to Ry
  // Additionally, there is a second factor of 2 which accounts for the 
  // factor of 2 difference in mass between Rydberg and Hartree atomic units
  // the JDFTx force constants already include the atomic masses
  forceConstants = forceConstants * 2. * 2.; 

  Eigen::MatrixXd bravaisVectors(3,nCells); 
  // convert cell map to cartesian coordinates 
  for(int iR = 0; iR<nCells; iR++) {

    // use cellMap to get R vector indices
    std::vector<int> Rcrys = cellMap[iR];

    // convert from crystal coordinate indices to cartesian coordinates
    // to match Phoebe conventions 
    Eigen::Vector3d Rcart = Rcrys[0] * directUnitCell.row(0) + Rcrys[1] * directUnitCell.row(1) +  Rcrys[2] * directUnitCell.row(2);
    bravaisVectors(0,iR) = Rcart(0); 
    bravaisVectors(1,iR) = Rcart(1); 
    bravaisVectors(2,iR) = Rcart(2); 

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

  // Now we do postprocessing
  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, bornCharges);
  crystal.print();

  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed harmonic JDFTx files.\n" << std::endl;
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
  return {crystal, dynamicalMatrix};
}
