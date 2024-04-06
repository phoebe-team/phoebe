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
  std::string directory = "/mnt/ceph/users/jcoulter/11.Si-hydro/1.jdftx/";

  std::string fileName = directory + "phonon.out";
  if (fileName.empty())
    Error("Must provide a totalE.out file name");

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
  Eigen::Vector3i qCoarseGrid;

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
    if(line.find("supercell") != std::string::npos
        && line.find("\\") != std::string::npos) {

      auto mesh = split(line, ' ');
      qCoarseGrid[0] = std::stoi(mesh[1]);
      qCoarseGrid[1] = std::stoi(mesh[2]);
      qCoarseGrid[2] = std::stoi(mesh[3]);
    }

    // check if the line contains ionic positions
    if (line.find("initial-state ") != std::string::npos) {
      // next lines contain ion positions in lattice coords
      // and atomic species
      std::cout << "found forces-output-coords " << line << std::endl;
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

  // ready final data structures
  int numElements = speciesNames.size();
  int numAtoms = tempPositions.size();

  Eigen::VectorXd speciesMasses(numElements);
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);

  PeriodicTable periodicTable;
  for (int i = 0; i < numElements; i++)
  {  speciesMasses[i] = periodicTable.getMass(speciesNames[i]);
     std::cout << "mass atom " << i << " " << speciesMasses[i] << std::endl;
  }
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
    Error("*.phononCellMap file not found in " + directory + ".");
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
  if (fileName.empty()) {
    Error("*.phononOmegaSq not found at " + fileName);
  }

  // open input file
  infile.open(fileName, std::ios::in | std::ios::binary);

  if (not infile.is_open()) {
    Error("D2 file not found");
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

  // reshape the force constants to match the expected phoebe format 
  auto mapped_t = Eigen::TensorMap<Eigen::Tensor<double, 7>>(&buffer[0], 
                                          3, 3, 
                                          qCoarseGrid[0], qCoarseGrid[1], qCoarseGrid[2], 
                                          numAtoms, numAtoms);
  // force constants matrix 
  Eigen::Tensor<double, 7> forceConstants = mapped_t;

  std::cout << std::scientific;
  int counter = 0;
  for(int iR = 0; iR<nCells; iR++) {

    // use cellMap to get R vector indices
    std::vector<int> Rcrys = cellMap[iR];

    // index this cell to the position in the supercell
    //int m1 = mod((Rcrys[0] + 1), qCoarseGrid(0));
    //if (m1 <= 0) m1 += qCoarseGrid(0);

    //int m2 = mod((Rcrys[1] + 1), qCoarseGrid(1));
    //if (m2 <= 0) m2 += qCoarseGrid(1);

    //int m3 = mod((Rcrys[2] + 1), qCoarseGrid(2));
    //if (m3 <= 0) m3 += qCoarseGrid(2);

    //m1 += -1; m2 += -1; m3 += -1;

    // convert from crystal coordinate indices to cartesian coordinates
    // to match Phoebe conventions 
    Eigen::Vector3d Rcart = Rcrys[0] * directUnitCell.row(0) + Rcrys[1] * directUnitCell.row(1) +  iRcrysR[2] * directUnitCell.row(2);
    //std::cout << "icell conversion " << iR[0] << iR[1] << iR[2] << " " << m1 << " " << m2 << " " << m3 << std::endl;
    //std::cout << "iR vs R " << iR[0] << " " << iR[1] << " " << iR[2] << "  " << R[0] << " " << R[1] << " " << R[2] << std::endl;

    // loop over nMode1
    // I think this should go nAtom1x, nAtom1y..
    for(int ia1 = 0; ia1<numAtoms; ia1++) {
      for(int ic1 = 0; ic1<3; ic1++) {

        // loop over nMode2
        for(int ia2 = 0; ia2<numAtoms; ia2++) {
          for(int ic2 = 0; ic2<3; ic2++) {
/*
           // need to discard these buffer elements, too
           if(iR[0] < 0 || iR[1] < 0 || iR[2] < 0) {
              // lets try only the positive quadrant
              counter++;
            }
           else {
            int m1 = iR[0];
            int m2 = iR[1];
            int m3 = iR[2];
*/
            // units are in hartree/bohr^2, convert to Ry/bohr^2
            forceConstants(ic1,ic2,m1,m2,m3,ia1,ia2) = buffer[counter]*2.;
            counter++;
            //std::cout << forceConstants(ic1,ic2,m1,m2,m3,ia1,ia2) << " ";
//           }

          }
        }
        //std::cout << std::endl;
      }
    }
    //std::cout << std::endl;
  }
  infile.close();

  // Now we do postprocessing
  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses);
  crystal.print();

  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error("qCoarseGrid smaller than zero");
  }
  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed harmonic JDFTx files.\n" << std::endl;
  }

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, bornCharges,
                           forceConstants, context.getSumRuleFC2());

  return {crystal, dynamicalMatrix};
}
