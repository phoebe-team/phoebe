#include <algorithm> // to use .remove_if
#include <cmath>     // round()
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "periodic_table.h"
#include "phonopy_input_parser.h"
#include "points.h"
#include "utilities.h"

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

// look up the index of an R vector in the cell positions list
int findRIndex(Eigen::MatrixXd &cellPositions2, Eigen::Vector3d &position2) {
  int ir2 = -1;
  for (int i = 0; i < cellPositions2.cols(); i++) {
    if ((position2 - cellPositions2.col(i)).squaredNorm() < 1.e-6) {
      ir2 = i;
      return ir2;
    }
  }
  if (ir2 == -1) {
    Error("Developer error: force constant R vector index not found in R vector list.");
  }
  return ir2;
}

std::tuple<Crystal, PhononH0> PhonopyParser::parsePhHarmonic(Context &context) {

  Kokkos::Profiling::pushRegion("parsePhHarmonic");

  // Here read the real space dynamical matrix of inter-atomic force constants
  if (mpi->mpiHead()) {
    std::cout << "Using harmonic force constants from phonopy." << std::endl;
  }

  double distanceConversion = 1. / distanceBohrToAng;

  // Read phono3py_disp.yaml file
  // ===================================================
  // Read in unit cell crystal information from phono3py_disp.yaml
  // This file is created at the start of a phono3py run, and contains
  // information specifying if fc2 has the same superCell dims as fc3

  // open input file
  std::string fileName = context.getPhonopyDispFileName();
  std::ifstream infile(fileName);
  std::string line;

  if (fileName.empty()) {
    Error("Phonopy required file phonopyDispFileName (phono3py_disp.yaml) "
          "file not specified in input file.");
  }
  if (not infile.is_open()) {
    Error("Phonopy required file phonopyDispFileName (phono3py_disp.yaml) "
          "not found at " +
          fileName);
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  // read in the dimension information.
  // we have to do this first, because we need to use this info
  // tllPositions2 the below data storage, and decide how we should look
  // for supercell information.
  // --------------------------------------------------------
  Eigen::Vector3i qCoarseGrid = {0,0,0};
  std::string supercellSearchString = "supercell:";
  while (std::getline(infile, line)) {

    // In the case where force constants where generated with different
    // superCells for fc2 and fc3, the label we need is dim_fc2.
    // Otherwise, if both are the same, it's just dim.
    if (line.find("phonon_supercell_matrix:") != std::string::npos) {
      for(int i : {0, 1, 2}) {
        std::getline(infile, line);
        std::vector<std::string> tok = tokenize(line);
        qCoarseGrid[i] = std::stoi(tok[2+i]);
        for(int j : {0, 1, 2}) {
          if(j != i && std::stoi(tok[2+j]) != 0) {
            Error("The phonopy cell you used has a non-diagonal supecell matrix.\n"
                "Phoebe presently doesn't know how to support this."
                "Revisit the ph transport tutorial\n"
                "and consider running the first step to generate a primitive cell before\n"
                "doing this calculation. Otherwise, contact the developers.");
          }
        }
      }
      supercellSearchString = "phonon_supercell:";
      break;
    }
    // this comes up first in the file, so if a different harmonic dim is present,
    // we'll overwrite it
    if (line.find("supercell_matrix:") != std::string::npos) {
      for(int i : {0, 1, 2}) {
        std::getline(infile, line);
        std::vector<std::string> tok = tokenize(line);
        qCoarseGrid[i] = std::stoi(tok[2+i]);
        for(int j : {0, 1, 2}) {
          if(j != i && std::stoi(tok[2+j]) != 0) {
            Error("The phonopy cell you used has a non-diagonal supecell matrix.\n"
                "Phoebe presently doesn't know how to support this."
                "Revisit the ph transport tutorial\n"
                "and consider running the first step to generate a primitive cell before\n"
                "doing this calculation. Otherwise, contact the developers.");
          }
        }
      }
    }
  }
  infile.clear();
  infile.seekg(0);

  if (qCoarseGrid(0) <= 0 || qCoarseGrid(1) <= 0 || qCoarseGrid(2) <= 0) {
    Error("Phonon super cell dims read as 0 or less.\n"
          "Something is wrong with your input.");
  }

  // read the rest of the file to find atomic positions,
  // lattice vectors, and species
  // --------------------------------------------------------
  Eigen::Matrix3d directUnitCell;
  std::vector<std::vector<double>> atomicPositionsVec;
  std::vector<int> atomicSpeciesVec;
  std::vector<std::string> speciesNames;
  PeriodicTable pt;
  bool foundUnitCell = false;
  // check that prim cell is the same and if not throw a warning
  bool foundPrimCell = false;
  Eigen::Matrix3d primUnitCell;

  int ilatt = 3;
  // Note: watch out, this is reading the unit cell from phono3py.
  // This could be different than the primitive cell, which can cause
  // issues in phono3py because of some conversions they do internally.
  // I throw a warning if I find prim and unit are not the same.
  while (infile) {
    getline(infile, line);
    // distance unit changes based on dft solver used
    if (line.find("length") != std::string::npos) {
      if (line.find("au") != std::string::npos) {
        distanceConversion = 1.; // distances already in Bohr
      }
    }

    if(foundUnitCell) {
     // if this line has a species, save it
     if (line.find(" symbol: ") != std::string::npos) {
       std::string temp =
           line.substr(line.find("symbol: ") + 8, line.find('#') - 13);
      // remove any trailing whitespaces
      temp.erase(std::remove_if(temp.begin(), temp.end(), ::isspace),
                 temp.end());
      if (std::find(speciesNames.begin(), speciesNames.end(), temp) ==
          speciesNames.end()) {
        speciesNames.push_back(temp);
      }
      // save the atom number of this species
      atomicSpeciesVec.push_back(
          std::find(speciesNames.begin(), speciesNames.end(), temp) -
          speciesNames.begin());
      }
      // if this is a cell position, save it
      if (line.find("coordinates: ") != std::string::npos) {
        std::vector<double> position(3);
        std::vector<std::string> tok = tokenize(line);
        position[0] = std::stod(tok[2]);
        position[1] = std::stod(tok[3]);
        position[2] = std::stod(tok[4]);
        atomicPositionsVec.push_back(position);
      }
      // parse lattice vectors
      if (ilatt < 3) { // count down lattice lines
        std::vector<std::string> tok = tokenize(line);
        directUnitCell(ilatt, 0) = std::stod(tok[2]);
        directUnitCell(ilatt, 1) = std::stod(tok[3]);
        directUnitCell(ilatt, 2) = std::stod(tok[4]);
        ilatt++;
      }
      if (line.find("lattice:") != std::string::npos) {
        ilatt = 0;
      }
      // this signals we are done reading cell info
      if (line.empty()) {
        break;
      }
    }
    if(foundPrimCell) {

      if (ilatt < 3) { // count down lattice lines
        std::vector<std::string> tok = tokenize(line);
        primUnitCell(ilatt, 0) = std::stod(tok[2]);
        primUnitCell(ilatt, 1) = std::stod(tok[3]);
        primUnitCell(ilatt, 2) = std::stod(tok[4]);
        ilatt++;
        // done prim cell read in
        if(ilatt == 3) foundPrimCell = false;
      }
      if (line.find("lattice:") != std::string::npos &&
                         line.find("recip") == std::string::npos) {
        ilatt = 0;
      }
    }
    if (line.find("unit_cell:") != std::string::npos) {
      foundUnitCell = true;
    }
    if (line.find("primitive_cell:") != std::string::npos) {
      foundPrimCell = true;
    }
  }

  // check that unit and prim cell match, otherwise throw a warning
  if(directUnitCell != primUnitCell) {
    Warning("Your unit cell does not match phonopy's primitive cell.\n"
        "This can cause errors in Phoebe, as it complicates phonopy's output.\n"
        "To ensure there are not errors, we recommend you go to the documentation and\n"
        "follow the instruction to transform your unit cell to phonopy's prim cell.\n"
        "https://phoebe.readthedocs.io/en/develop/tutorials/phononTransport.html#step-2-calculation-of-force-constants\n"
        "We recommend against using --pa in phonopy."); // TODO should be error?
  }

  // read the rest of the file to get FC2 superCell positions
  // --------------------------------------------------------
  std::vector<std::vector<double>> supPositionsVec;
  Eigen::Matrix3d supLattice;
  ilatt = 3;
  bool readSupercell = false;
  while (infile) {

    getline(infile, line);

    // read all the lines after we see the flag for the supercell
    if (line.find(supercellSearchString) != std::string::npos) {
      readSupercell = true;
    }
    // if this is a cell position, save it
    if (line.find("coordinates: ") != std::string::npos && readSupercell) {
      std::vector<double> position(3);
      std::vector<std::string> tok = tokenize(line);
      position[0] = std::stod(tok[2]);
      position[1] = std::stod(tok[3]);
      position[2] = std::stod(tok[4]);
      supPositionsVec.push_back(position);
    }
    if (ilatt < 3 && readSupercell) { // count down lattice lines
      // convert from angstrom to bohr
      std::vector<std::string> tok = tokenize(line);
      supLattice(ilatt, 0) = std::stod(tok[2]);
      supLattice(ilatt, 1) = std::stod(tok[3]);
      supLattice(ilatt, 2) = std::stod(tok[4]);
      ilatt++;
    }
    if (line.find("lattice:") != std::string::npos && readSupercell) {
      ilatt = 0;
    }
  }
  infile.close();

  // number of atoms in the supercell for later use
  int numSupAtoms = supPositionsVec.size();

  // set number of unit cell atoms
  int numAtoms = atomicPositionsVec.size();

  // convert distances to Bohr
  supLattice *= distanceConversion;
  directUnitCell *= distanceConversion;

  // convert std::vectors to Eigen formats required by the
  // next part of phoebe
  Eigen::VectorXi atomicSpecies(numAtoms);
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::MatrixXd supPositions(numSupAtoms, 3);
  // copy to Eigen format
  for ( int n = 0; n < numAtoms; n++) {
    for (int i : {0, 1, 2}) {
        atomicPositions(n,i) = atomicPositionsVec[n][i];
        atomicSpecies(n) = atomicSpeciesVec[n];
    }
  }
  for ( int n = 0; n < numSupAtoms; n++) {
    for (int i : {0, 1, 2}) {
        supPositions(n,i) = supPositionsVec[n][i];
    }
  }

  // Process the information that has been read in
  // =================================================
  // calculate species mass values (in Ry) for later use
  Eigen::VectorXd speciesMasses(speciesNames.size());
  int count = 0;
  for (const auto& i : speciesNames) {
    speciesMasses[count] = pt.getMass(i) * massAmuToRy;
    count++;
  }

  // convert superCell positions to cartesian, in bohr (supLattice in bohr)
  for (int i = 0; i < numSupAtoms; i++) {
    Eigen::Vector3d temp = supPositions.row(i);
    Eigen::Vector3d temp2 = supLattice.transpose() * temp;
    supPositions.row(i) = temp2;
  }

  // convert unit cell positions to cartesian, in bohr
  for (int i = 0; i < numAtoms; i++) {
    Eigen::Vector3d temp = atomicPositions.row(i);
    // lattice already in Bohr
    Eigen::Vector3d temp2 = directUnitCell.transpose() * temp;
    atomicPositions.row(i) = temp2;
  }

  // Determine the list of possible R2, R3 vectors
  // the distances from the unit cell to a superCell
  // nCells here is the number of unit cell copies in the superCell
  int nCells = qCoarseGrid.prod();
  Eigen::MatrixXd cellPositions2(3, nCells);
  cellPositions2.setZero();
  for (int iCell = 0; iCell < nCells; iCell++) {
    // find the non-WS cell R2 vectors which are
    // position of atomPosSuperCell - atomPosUnitCell = R
    cellPositions2.col(iCell) = supPositions.row(iCell) - supPositions.row(0);
  }

  // If hasDielectric, look for BORN file
  // ============================================================
  // the BORN file contains the dielectric matrix on the first line,
  // and the BECs of unique atoms on the following lines
  Eigen::Matrix3d dielectricMatrix;
  dielectricMatrix.setZero();
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  // the below code will parse the BORN file
  fileName = context.getPhonopyBORNFileName();
  // skip reading in born charges if file name is not set
  if (!fileName.empty()) {

    infile.clear();
    infile.open(fileName);

    if (!infile.is_open()) {
      Error("BORN file " + fileName + " cannot be read.");
    }
    if(mpi->mpiHead()) {
      std::cout << "\nReading in the phonopy BORN file." << std::endl;
    }

    // in current versions of phonopy, the first line either contains the
    // unit conversion or the "default conversion". In old versions, it was a
    // comment containing atom info.
    // we're ignoring this for now, as these conversions do not appear right for us.
    // in fact, BECs are almost always in units of e, so that had better be what the
    // user uses. For now, we read this line to skip it
    getline(infile,line);

    bool readDielectric = false;
    int iat = 0;
    while (getline(infile,line)) {

      // make sure it's not a comment
      if(line.find("#") != std::string::npos) continue;
      // make sure it's not blank
      if(line.find_first_not_of(' ') == std::string::npos) continue;

      // the first non-comment line in the file is the dielectric matrix
      if(!readDielectric) {
        std::istringstream iss2(line);
        iss2 >> dielectricMatrix(0,0) >> dielectricMatrix(0,1) >>
          dielectricMatrix(0,2) >> dielectricMatrix(1,0) >> dielectricMatrix(1,1) >>
          dielectricMatrix(1,2) >> dielectricMatrix(2,0) >> dielectricMatrix(2,1) >>
          dielectricMatrix(2,2);
        readDielectric = true;
      }
      else {  // parse the born charges in the rest of the file
        std::vector<std::string> tok = tokenize(line);
         for(int i = 0; i < 3; i++) {
           for(int j = 0; j < 3; j++) {
              bornCharges(iat,i,j) = std::stod(tok[i*3 + j]);
           }
         }
        iat++;
      }
    }
    // print the charge matrices
    if(mpi->mpiHead()) {

      std::cout << "Dielectric matrix read as:" << std::endl;
      std::cout << dielectricMatrix << '\n' << std::endl;
      std::cout << "Born effective charges read as:" << std::endl;
      for (int idx = 0; idx < iat; idx++) {
        std::cout << speciesNames[idx] << " ";
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            std::cout << bornCharges(idx,i,j) << " ";
          }
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

// Parse the fc2.hdf5 file and read in the dynamical matrix
// ==========================================================
#ifndef HDF5_AVAIL
  Error(
      "Phono3py HDF5 output cannot be read if Phoebe is not built with HDF5.");
#else

  // set up buffer to read entire matrix
  // have to use this because the phono3py data is shaped as a
  // 4 dimensional array, and eigen tensor is not supported by highFive
  std::vector<std::vector<std::vector<std::vector<double>>>> ifc2;
  std::vector<int> cellMap;
  HighFive::FixedLenStringArray<14> unitVec;
  std::string unit;

  fileName = context.getPhFC2FileName();
  if (fileName.empty()) {
    Error("Phonopy required file phFC2FileName (fc2.hdf5) file not "
          "specified in input file.");
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  try {
    // Open the hdf5 file
    HighFive::File file(fileName, HighFive::File::ReadOnly);

    // Set up hdf5 datasets
    HighFive::DataSet difc2 = file.getDataSet("/force_constants");
    HighFive::DataSet dCellMap = file.getDataSet("/p2s_map");
    // read in the ifc2 data
    difc2.read(ifc2);
    dCellMap.read(cellMap);

    // check that cell map matches number of atoms
    if(int(cellMap.size()) != numAtoms) {
      Error("Developer error: p2s_map from phono3py does not match numAtoms."
                "\nyaml file and HDF5 file are somehow mismatched.");
    }

    // unfortunately it appears this is not in some fc files...
    // default to ev/Ang^2
    try {
      HighFive::DataSet dConversion = file.getDataSet("/physical_unit");
      dConversion.read(unitVec);
      unit = unitVec[0];
    } catch (std::exception &error) {
      if(mpi->mpiHead()) {
        std::cout << "\nPhonopy fc file did not include units. "
         << "\nThis is likely ok, defaulting to eV/angstrom^2."
         << "\nHowever, you should check to be sure the magnitude of your"
         << " phonon frequencies is sensible.\n" << std::endl;
      }
      unit = "eV/angstrom^2";
    }

  } catch (std::exception &error) {
    if(mpi->mpiHead()) std::cout << error.what() << std::endl;
    Error("Issue reading fc2.hdf5 file. Make sure it exists at " + fileName +
          "\n and is not open by some other persisting processes.");
  }

  Eigen::Tensor<double, 7> forceConstants(3, 3, qCoarseGrid[0], qCoarseGrid[1],
                                          qCoarseGrid[2], numAtoms, numAtoms);

  // if the force constants are compact format, the first two
  // dimensions will not be the same (one will be nprimAtoms, other nsupAtoms)
  bool compact = false;
  if(ifc2.size() != ifc2[0].size()) compact = true;

  // phonopy force constants are in ev/ang^2, convert to atomic
  double conversion = 1;
  if(unit.find("eV/ang") != std::string::npos) {
    conversion = pow(distanceBohrToAng, 2) / energyRyToEv;
  }

  // for the second atom, we must loop over all possible
  // cells in the superCell containing copies of these
  // unit cell atoms
  for (int r3 = 0; r3 < qCoarseGrid[2]; r3++) {
    for (int r2 = 0; r2 < qCoarseGrid[1]; r2++) {
      for (int r1 = 0; r1 < qCoarseGrid[0]; r1++) {

        // NOTE we do this because phonopy has an
        // "old" and "new" format for superCell files,
        // and there's not an eay way for us to tell which
        // one a user might have loaded in. Therefore,
        // we can't intelligently guess the ordering of
        // atoms in the superCell, and instead use R
        // to find their index.

        // build the R vector associated with this
        // cell in the superCell
        Eigen::Vector3d R;
        R = r1 * directUnitCell.row(0) + r2 * directUnitCell.row(1) +
            r3 * directUnitCell.row(2);

        // use the find cell function to determine
        // the index of this cell in the list of
        // R vectors (named cellPositions from above)
        int ir = findRIndex(cellPositions2, R);

        // loop over the first atoms. Because we consider R1=0,
        // these are only primitive unit cell atoms.
        for (int iat = 0; iat < numAtoms; iat++) {
          for (int jat = 0; jat < numAtoms; jat++) {

            // Need to convert to superCell indices.
            // Atoms in superCell are ordered so that there is a
            // unit cell atom followed by numUnitCell-in-superCell-#-of-atoms,
            // which are representations of the unit cell atom in other cells
            // we find this by using cellMap to tell us where the
            // first atom of this type is, then adding ir, which tells us
            // which cell it's in.
            //
            int isAt, jsAt;
            if(!compact) {
              // We must put an offset of iR onto the first index, because
              // the format in phono3py is D(R',R=0)
              isAt = cellMap[iat] + ir;
              jsAt = cellMap[jat];
            }
            else {
              // if compact, now it appears notation is D(R=0,R')
              jsAt = cellMap[iat] + ir;
              isAt = jat;
            }
            // loop over cartesian directions
            for (int ic : {0, 1, 2}) {
              for (int jc : {0, 1, 2}) {

                // here, cellMap tells us the position of this
                // unit cell atom in the superCell of phonopy
                forceConstants(ic, jc, r1, r2, r3, iat, jat) =
                    ifc2[isAt][jsAt][ic][jc] * conversion;
              }
            }
          }
        }
      }
    }
  }

  if (mpi->mpiHead()) {
    std::cout << "Successfully parsed harmonic phonopy files.\n" << std::endl;
  }

  // Now we do postprocessing
  // must transpose the lattice vectors before passing to crystal,
  // as crystal uses a different format than phonopy does.
  directUnitCell.transposeInPlace();

  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses);
  crystal.print();
  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, bornCharges,
                           forceConstants, context.getSumRuleFC2());

  Kokkos::Profiling::popRegion();
  return std::make_tuple(crystal, dynamicalMatrix);
#endif
}
