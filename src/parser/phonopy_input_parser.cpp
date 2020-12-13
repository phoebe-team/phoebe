#include <algorithm> // to use .remove_if
#include <fstream>
#include <iomanip> // to declare istringstream
#include <iostream>
#include <math.h>   // round()
#include <stdlib.h> // abs()
#include <string>
#include <vector>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "particle.h"
#include "periodic_table.h"
#include "phonopy_input_parser.h"
#include "utilities.h"
#include "full_points.h"

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

long findRIndex(Eigen::MatrixXd &cellPositions2, Eigen::Vector3d &position2) {
  long ir2 = -1;
  for (int i = 0; i < cellPositions2.cols(); i++) {
    if ((position2 - cellPositions2.col(i)).norm() < 1.e-12) {
      ir2 = i;
      return ir2;
    }
  }
  if (ir2 == -1) {
    Error e("index not found");
  }
  return ir2;
}

std::tuple<Crystal, PhononH0> PhonopyParser::parsePhHarmonic(Context &context) {
  //  Here we read the dynamical matrix of interatomic force constants
  //    in real space.

  // we have to read information from the disp_fc2.yaml file, 
  // which has the fc2 supercell regardless of whether or not forces 
  // were generated with different fc2 vs fc3 supercells.
  std::string fileName = context.getPhD2FileName();
  if (fileName == "") {
    Error e("Must provide a disp_fc3.yaml (if p3py force constants "
        "were generated with same dim for fc2 and fc3 supercells) "
        "or disp_fc2.yaml (if supercell dims were different).", 1);
  }

  // open input file
  std::ifstream infile(fileName);
  std::string line;

  if (not infile.is_open()) {
    Error e("disp_fc2.yaml/disp_fc3.yaml file not found", 1);
  }

  // first line will always be natoms in supercell
  std::getline(infile, line);
  int numSupAtoms = std::stoi(line.substr(line.find(" ") ,line.back()));

  // read the rest of the file to get supercell positions
  Eigen::MatrixXd supPositions(numSupAtoms, 3);
  Eigen::MatrixXd supLattice(3, 3);
  int ilatt = 3;
  int ipos = 0;
  while(infile) {
    getline(infile, line);
    // if this is a cell position, save it
    if(line.find("position: ") != std::string::npos) {
      std::string temp = line.substr(14,57); // just the positions
      int idx1 = temp.find(",");
      supPositions(ipos,0) = std::stod(temp.substr(0,idx1));
      int idx2 = temp.find(",", idx1+1);
      supPositions(ipos,1) = std::stod(temp.substr(idx1+1,idx2));
      supPositions(ipos,2) = std::stod(temp.substr(idx2+1));
      ipos++;
    }
    if(ilatt < 3) { // count down lattice lines 
      // convert from angstrom to bohr
      std::string temp = line.substr(5,62); // just the elements
      int idx1 = temp.find(",");
      supLattice(ilatt,0) = std::stod(temp.substr(0,idx1))/distanceBohrToAng;
      int idx2 = temp.find(",", idx1+1);
      supLattice(ilatt,1) = std::stod(temp.substr(idx1+1,idx2))/distanceBohrToAng;
      supLattice(ilatt,2) = std::stod(temp.substr(idx2+1))/distanceBohrToAng;
      ilatt++;
    }
    if(line.find("lattice:") != std::string::npos) {
      ilatt = 0;
    }
  }
  infile.close();

  // ===================================================
  // now, we read in unit cell crystal information from phono3py.yaml
  fileName = context.getPhD2FileName();
  if (fileName == "") {
    Error e("Must provide a phono3py.yaml file.", 1);
  }

  // open input file
  std::ifstream infile2(fileName);

  // open input file
  if (not infile2.is_open()) {
    Error e("phono3py.yaml file not found", 1);
  }
 
  // read in the dimension information. 
  // we have to do this first, because we need to use this info
  // to allocate the below data storage.
  Eigen::Vector3i qCoarseGrid;
  while(infile2) {
    getline(infile2, line);

    // In the case where force constants where generated with different
    // supercells for fc2 and fc3, the label we need is dim_fc2. 
    // Otherwise, if both are the same, it's just dim.
    if(line.find("dim_fc2: ") != std::string::npos) {
      std::string temp = line.substr(line.find("\""), line.find("\"\n"));
      std::istringstream iss(line);
      iss >> qCoarseGrid[0] >> qCoarseGrid[1] >> qCoarseGrid[2];
      break;
    }
    if(line.find("dim: ") != std::string::npos) {
      std::string temp = line.substr(line.find("\""), line.find("\"\n"));
      std::istringstream iss(line);
      iss >> qCoarseGrid[0] >> qCoarseGrid[1] >> qCoarseGrid[2];
      break;
    }
  }
  // set number of unit cell atoms
  int numAtoms = numSupAtoms/(qCoarseGrid[0]*qCoarseGrid[1]*qCoarseGrid[0]);

  // read the rest of the file to find atomic positions, 
  // lattice vectors, and species
  Eigen::Matrix3d directUnitCell;

  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);

  std::vector<std::string> speciesNames;
  std::vector<double> allSpeciesMasses;

  ilatt = 3;
  ipos = 0;
  //bool readSuperCell = false;
  //bool readUnitCell = true;
  //bool waitForSupercell = false;
  // TODO watchout, this is reading the primitive cell from phono3py. 
  // we might want to read in the unit cell, which could be different
  // because of some conversions they do internally. 
  // Unit cell is also written to this fine in the same way as read
  // below.
  while(infile) {
    getline(infile, line);

    //if(readUnitCell) { 
      // if this line has a species, save it 
      if(line.find("symbol: ") != std::string::npos) {
        speciesNames.push_back(line.substr(13,line.find("#")-1));
      }
      // if this line has a mass, save it 
      if(line.find("mass: ") != std::string::npos) {
        allSpeciesMasses.push_back(std::stod(line.substr(10))); // TODO convert to ry?
      }
      // if this is a cell position, save it
      if(line.find("coordinates: ") != std::string::npos) {
        std::string temp = line.substr(19,59); // just the positions
        int idx1 = temp.find(",");
        atomicPositions(ipos,0) = std::stod(temp.substr(0,idx1));
        int idx2 = temp.find(",", idx1+1);
        atomicPositions(ipos,1) = std::stod(temp.substr(idx1+1,idx2));
        atomicPositions(ipos,2) = std::stod(temp.substr(idx2+1));
        ipos++;
      }
      // parse lattice vectors
      if(ilatt < 3) { // count down lattice lines 
        std::string temp = line.substr(5,62); // just the elements
        int idx1 = temp.find(",");
        directUnitCell(ilatt,0) = std::stod(temp.substr(0,idx1));
        int idx2 = temp.find(",", idx1+1);
        directUnitCell(ilatt,1) = std::stod(temp.substr(idx1+1,idx2));
        directUnitCell(ilatt,2) = std::stod(temp.substr(idx2+1));
        ilatt++;
      }
      if(line.find("lattice:") != std::string::npos) {
        ilatt = 0;
      }
      // this signals we are done reading primitive cell info
      // and want to wait until we see the supercell lines
      if(line.find("unit_cell:") != std::string::npos) {
        break;
        //readUnitCell = false;
      }
    //}
    //if(readSuperCell) { 
      
    //}
    // swap from reading unit cell to supercell
    //if(line.find("supercell:") != std::string::npos) {
    //  readUnitCell = false;
    //  readSuperCell = true;
   // }
  }
  infile2.close();

  // take only the unique mass values for later use
  auto iter = std::unique(allSpeciesMasses.begin(), allSpeciesMasses.end());
  allSpeciesMasses.resize( std::distance(allSpeciesMasses.begin(),iter));
  Eigen::VectorXd speciesMasses = Eigen::VectorXd::Map(allSpeciesMasses.data(), allSpeciesMasses.size());

  // convert supercell positions to cartesian, in bohr
  for(int i = 0; i<numSupAtoms; i++) {
    Eigen::Vector3d temp(supPositions(i,0),supPositions(i,1),supPositions(i,2));
    Eigen::Vector3d temp2 = supLattice * temp;
    supPositions(i,0) = temp2(0);
    supPositions(i,1) = temp2(1);
    supPositions(i,2) = temp2(2);
  }

  // convert unit cell positions to cartesian, in bohr
  for(int i = 0; i<numAtoms; i++) {
    Eigen::Vector3d temp(
        atomicPositions(i,0),atomicPositions(i,1),atomicPositions(i,2));
    Eigen::Vector3d temp2 =  directUnitCell * temp;
    temp2 = temp2 / distanceBohrToAng; // lattice vectors are in angstrom
    atomicPositions(i,0) = temp2(0);
    atomicPositions(i,1) = temp2(1);
    atomicPositions(i,2) = temp2(2);
  }

  // build the atomicSpecies list 
  // this is a list of integers specifying which species
  // number each element is 
  //std::vector<std::string> species;
  for(int i = 0; i<numAtoms; i++) {
    //auto speciesIdx = std::find(species.begin(), species.end(), species[i]);
    atomicSpecies(i) = std::find(speciesNames.begin(), speciesNames.end(), speciesNames[i]) - speciesNames.begin();
    // species was not in the list
    //if(std::find(species.begin(), species.end(), speciesNames[i]) == species.end()) {
    //  species.push_back(speciesNames[i]);
    //}
  }

  // Determine the list of possible R2, R3 vectors
  // the distances from the unit cell to a supercell 
  // nCells here is the number of unit cell copies in the supercell
  int nCells = qCoarseGrid[0]*qCoarseGrid[1]*qCoarseGrid[2];
  Eigen::MatrixXd cellPositions2(3, nCells);
  cellPositions2.setZero();
  for (int icell = 0; icell < nCells; icell++) {

      // find the non-WS cell R2 vectors which are 
      // position of atomPosSupercell - atomPosUnitCell = R
      cellPositions2(0,icell) = supPositions(icell,0) - supPositions(0,0);
      cellPositions2(1,icell) = supPositions(icell,1) - supPositions(0,1);
      cellPositions2(2,icell) = supPositions(icell,2) - supPositions(0,2);
  }

  //  Read if hasDielectric
  bool hasDielectric = false; // TODO for now, we just say no dielectric
  Eigen::Matrix3d dielectricMatrix;
  dielectricMatrix.setZero();
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  // Parse the fc2.hdf5 file and read in the dynamical matrix 
  #ifndef HDF5_AVAIL
    Error e("Phono3py HDF5 output cannot be read if Phoebe is not built with HDF5.");
    //return void;
  #else

  // now we parse the coarse q grid
  fileName = context.getPhD2FileName();
  if (fileName == "") {
    Error e("Must provide a D2 file name, like fc2.hdf5", 1);
  }

  // Open the hdf5 file
  HighFive::File file(fileName, HighFive::File::ReadOnly);

  // Set up hdf5 datasets
  HighFive::DataSet difc2 = file.getDataSet("/fc2");
  HighFive::DataSet dcellMap = file.getDataSet("/p2s_map");

  // set up buffer to read entire matrix
  // have to use this because the phono3py data is shaped as a
  // 4 dimensional array, and eigen tensor is not supported by highFive
  std::vector<std::vector<std::vector<std::vector<double>>>> ifc2;
  std::vector<int> cellMap;

  // read in the ifc3 data
  difc2.read(ifc2);
  dcellMap.read(cellMap);

  Eigen::Tensor<double, 7> forceConstants(
      3, 3, qCoarseGrid[0], qCoarseGrid[1], qCoarseGrid[2], numAtoms, numAtoms);

  // for the second atom, we must loop over all possible
  // cells in the supercell containing copies of these 
  // unit cell atoms
  for (int r3 = 0; r3 < qCoarseGrid[2]; r3++) {
    for (int r2 = 0; r2 < qCoarseGrid[1]; r2++) {
      for (int r1 = 0; r1 < qCoarseGrid[0]; r1++) {

        // NOTE we do this because phonopy has an 
        // "old" and "new" format for supercell files, 
        // and there's not an eay way for us to tell which 
        // one a user might have loaded in. Therefore, 
        // we can't intelligently guess the ordering of 
        // atoms in the supercell, and instead use R
        // to find their index. 

        // build the R vector associated with this 
        // cell in the supercell
        Eigen::Vector3d R;
        R = r1 * directUnitCell.row(0) +
            r2 * directUnitCell.row(1) +
            r3 * directUnitCell.row(2);

        // use the find cell function to determine
        // the index of this cell in the list of 
        // R vectors (named cellPositions from above)
        int ir = findRIndex(cellPositions2, R); 

        // loop over the first atoms. Because we consider R1=0, 
        // these are only primitive unit cell atoms.
        for (int iat = 0; iat < numAtoms; iat++) {
          for (int jat = 0; jat < numAtoms; jat++) {
   
            // Need to convert jat to supercell index 
            // Atoms in supercell are ordered so that
            // there is a unit cell atom followed by 
            // numUnitcell-in-supercell-#-of-atoms
            // TODO lets think of an atom in the 8th cell. 
            // the 8th cell. the 8th cell atoms are every idx 
            // 7, 15, 23, 31.... that's ir + iat*numAtom.
            int jsat = ir + numAtoms * jat; 

            // loop over cartesian directions
            for (int ic : {0,1,2}) {
              for (int jc : {0,1,2}) {

                // here, cellMap tells us the position of this
                // unit cell atom in the supercell of phonopy
                forceConstants(ic, jc, r1, r2, r3, iat, jat) = 
                      ifc2[cellMap[iat]][jsat][ic][jc];

              }
            }
          }
        }
      }
    }
  }
  #endif

  // Now we do postprocessing
  long dimensionality = context.getDimensionality();
  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses, dimensionality);

  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error e("qCoarseGrid smaller than zero", 1);
  }

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, bornCharges,
                           forceConstants, context.getSumRuleD2());

  return {crystal, dynamicalMatrix};
};
