#include <algorithm>// to use .remove_if
#include <cmath>    // round()
#include <cstdlib>  // abs()
#include <fstream>
#include <string>
#include <vector>

#include "constants.h"
#include "eigen.h"
#include "exceptions.h"
#include "particle.h"
#include "periodic_table.h"
#include "points.h"
#include "pugixml.hpp"
#include "qe_input_parser.h"
#include "utilities.h"

void getQELattice(const int iBravais, Eigen::VectorXd &celldm,
                  Eigen::Matrix3d &unitCell) {
  //  sets up the crystallographic vectors a1, a2, and a3.
  //
  //  iBravais is the structure index:
  //    1  cubic P (sc)
  //    2  cubic F (fcc)
  //    3  cubic I (bcc)
  //    4  hexagonal and trigonal P
  //    5  trigonal R, 3-fold axis c
  //    6  tetragonal P (st)
  //    7  tetragonal I (bct)
  //    8  orthorhombic P
  //    9  1-face (C) centered orthorhombic
  //   10  all face centered orthorhombic
  //   11  body centered orthorhombic
  //   12  monoclinic P (unique axis: c)
  //   13  one face (base) centered monoclinic
  //   14  triclinic P
  //  Also accepted:
  //    0  "free" structure
  //  -12  monoclinic P (unique axis: b)
  //   -3  cubic bcc with a more symmetric choice of axis
  //   -5  trigonal R, threefold axis along (111)
  //   -9  alternate description for base centered orthorhombic
  //  -13  one face (base) centered monoclinic (unique axis: b)
  //   91  1-face (A) centered orthorhombic
  //
  //  celldm are parameters which fix the shape of the unit cell
  //  volumeUnitCell is the unit-cell volume
  //
  //  NOTA BENE: all axis sets are right-handed
  //  Boxes for US PPs do not work properly with left-handed axis

  const double sr2 = 1.414213562373, sr3 = 1.732050807569;

  //  user-supplied lattice vectors

  Eigen::Vector3d a1, a2, a3;

  a1 = unitCell.col(0);
  a2 = unitCell.col(1);
  a3 = unitCell.col(2);

  if (iBravais == 0) {
    if (a1.norm() == 0. || a2.norm() == 0. || a3.norm() == 0.) {
      Error("wrong at for iBravais=0");
    }
    if (celldm(0) != 0.) {
      // input at are in units of aLat => convert them to a.u.
      unitCell *= celldm(0);
    } else {
      // input at are in atomic units: define celldm(1) from a1
      celldm(0) = a1.norm();
    }
  } else {
    a1.setZero();
    a2.setZero();
    a3.setZero();
  }

  if (celldm(0) <= 0.) {
    Error("wrong celldm(1)");
  }

  //  index of bravais lattice supplied

  if (iBravais == 1) {// simple cubic lattice
    a1(0) = celldm(0);
    a2(1) = celldm(0);
    a3(2) = celldm(0);
  } else if (iBravais == 2) {//     fcc lattice
    double term = celldm(0) / 2.;
    a1(0) = -term;
    a1(2) = term;
    a2(1) = term;
    a2(2) = term;
    a3(0) = -term;
    a3(1) = term;
  } else if (abs(iBravais) == 3) {// bcc lattice
    double term = celldm(0) / 2.;
    for (int ir = 0; ir < 3; ir++) {
      a1(ir) = term;
      a2(ir) = term;
      a3(ir) = term;
    }
    if (iBravais < 0) {
      a1(0) = -a1(0);
      a2(1) = -a2(1);
      a3(2) = -a3(2);
    } else {
      a2(0) = -a2(0);
      a3(0) = -a3(0);
      a3(1) = -a3(1);
    }
  } else if (iBravais == 4) {// hexagonal lattice
    if (celldm(2) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    double cbya = celldm(2);
    a1(0) = celldm(0);
    a2(0) = -celldm(0) / 2.;
    a2(1) = celldm(0) * sr3 / 2.;
    a3(2) = celldm(0) * cbya;

  } else if (abs(iBravais) == 5) {// trigonal lattice
    if (celldm(3) <= -0.5 || celldm(3) >= 1.) {
      Error("wrong celldm(4)", abs(iBravais));
    }

    double term1 = sqrt(1. + 2. * celldm(3));
    double term2 = sqrt(1. - celldm(3));

    if (iBravais == 5) {// threefold axis along c (001)
      a2(1) = sr2 * celldm(0) * term2 / sr3;
      a2(2) = celldm(0) * term1 / sr3;
      a1(0) = celldm(0) * term2 / sr2;
      a1(1) = -a1(0) / sr3;
      a1(2) = a2(2);
      a3(0) = -a1(0);
      a3(1) = a1(1);
      a3(2) = a2(2);
    } else if (iBravais == -5) {// threefold axis along (111)
      // Notice that in the cubic limit (alpha=90, celldm(4)=0,
      // term1=term2=1)
      // does not yield the x,y,z axis, but an equivalent rotated triplet:
      //   a/3 (-1,2,2), a/3 (2,-1,2), a/3 (2,2,-1)
      // If you prefer the x,y,z axis as cubic limit, you should modify
      // the definitions of a1(1) and a1(2) as follows:'
      // a1(1) = celldm(1)*(term1+2.0_dp*term2)/3.0_dp
      // a1(2) = celldm(1)*(term1-term2)/3.0_dp
      // (info by G. Pizzi and A. Cepellotti)
      a1(0) = celldm(0) * (term1 - 2. * term2) / 3.;
      a1(1) = celldm(0) * (term1 + term2) / 3.;
      a1(2) = a1(1);
      a2(0) = a1(2);
      a2(1) = a1(0);
      a2(2) = a1(1);
      a3(0) = a1(1);
      a3(1) = a1(2);
      a3(2) = a1(0);
    }
  } else if (iBravais == 6) {// tetragonal lattice
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    double cbya = celldm(2);
    a1(0) = celldm(0);
    a2(1) = celldm(0);
    a3(2) = celldm(0) * cbya;

  } else if (iBravais == 7) {// body centered tetragonal lattice
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    double cbya = celldm(2);
    a2(0) = celldm(0) / 2.;
    a2(1) = a2(0);
    a2(2) = cbya * celldm(0) / 2.;
    a1(0) = a2(0);
    a1(1) = -a2(0);
    a1(2) = a2(2);
    a3(0) = -a2(0);
    a3(1) = -a2(0);
    a3(2) = a2(2);
  } else if (iBravais == 8) {// Simple orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1);
    a3(2) = celldm(0) * celldm(2);
  } else if (abs(iBravais) == 9) {
    // One face (base) centered orthorhombic lattice  (C type)
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", abs(iBravais));
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", abs(iBravais));
    }
    if (iBravais == 9) {// old PW-scf description
      a1(0) = 0.5 * celldm(0);
      a1(1) = a1(0) * celldm(1);
      a2(0) = -a1(0);
      a2(1) = a1(1);
    } else {// alternate description
      a1(0) = 0.5 * celldm(0);
      a1(1) = -a1(0) * celldm(1);
      a2(0) = a1(0);
      a2(1) = -a1(1);
    }
    a3(2) = celldm(0) * celldm(2);
  } else if (iBravais == 91) {
    // One face(base)centered orthorhombic lattice (A type)
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1) * 0.5;
    a2(2) = -celldm(0) * celldm(2) * 0.5;
    a3(1) = a2(1);
    a3(2) = -a2(2);
  } else if (iBravais == 10) {// All face centered orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    a2(0) = 0.5 * celldm(0);
    a2(1) = a2(0) * celldm(1);
    a1(0) = a2(0);
    a1(2) = a2(0) * celldm(2);
    a3(1) = a2(0) * celldm(1);
    a3(2) = a1(2);
  } else if (iBravais == 11) {// Body centered orthorhombic lattice
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    a1(0) = 0.5 * celldm(0);
    a1(1) = a1(0) * celldm(1);
    a1(2) = a1(0) * celldm(2);
    a2(0) = -a1(0);
    a2(1) = a1(1);
    a2(2) = a1(2);
    a3(0) = -a1(0);
    a3(1) = -a1(1);
    a3(2) = a1(2);
  } else if (iBravais == 12) {
    // Simple monoclinic lattice, unique (i.e. orthogonal to a) axis: c
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    if (abs(celldm(3)) >= 1.) {
      Error("wrong celldm(4)", iBravais);
    }
    double sen = sqrt(1. - celldm(3) * celldm(3));
    a1(0) = celldm(0);
    a2(0) = celldm(0) * celldm(1) * celldm(3);
    a2(1) = celldm(0) * celldm(1) * sen;
    a3(2) = celldm(0) * celldm(2);
  } else if (iBravais == -12) {
    // Simple monoclinic lattice, unique axis: b (more common)
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", -iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", -iBravais);
    }
    if (abs(celldm(4)) >= 1.) {
      Error("wrong celldm(5)", -iBravais);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = celldm(0);
    a2(1) = celldm(0) * celldm(1);
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(2) = celldm(0) * celldm(2) * sen;
  } else if (iBravais == 13) {
    // One face centered monoclinic lattice unique axis c
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    if (abs(celldm(3)) >= 1.) {
      Error("wrong celldm(4)", iBravais);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = 0.5 * celldm(0);
    a1(2) = -a1(0) * celldm(2);
    a2(0) = celldm(0) * celldm(1) * celldm(2);
    a2(1) = celldm(0) * celldm(1) * sen;
    a3(0) = a1(0);
    a3(2) = -a1(2);
  } else if (iBravais == -13) {
    // One face centered monoclinic lattice unique axis b
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", -iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", -iBravais);
    }
    if (abs(celldm(4)) >= 1.) {
      Error("wrong celldm(5)", -iBravais);
    }
    double sen = sqrt(1. - celldm(4) * celldm(4));
    a1(0) = 0.5 * celldm(0);
    a1(1) = -a1(0) * celldm(1);
    a2(0) = a1(0);
    a2(1) = -a1(1);
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(2) = celldm(0) * celldm(2) * sen;
  } else if (iBravais == 14) {// Triclinic lattice
    if (celldm(1) <= 0.) {
      Error("wrong celldm(2)", iBravais);
    }
    if (celldm(2) <= 0.) {
      Error("wrong celldm(3)", iBravais);
    }
    if (abs(celldm(3)) >= 1.) {
      Error("wrong celldm(4)", iBravais);
    }
    if (abs(celldm(4)) >= 1.) {
      Error("wrong celldm(5)", iBravais);
    }
    if (abs(celldm(5)) >= 1.) {
      Error("wrong celldm(6)", iBravais);
    }
    double sigma = sqrt(1. - celldm(5) * celldm(5));
    double term =
        (1. + 2. * celldm(3) * celldm(4) * celldm(5) - celldm(3) * celldm(3) - celldm(4) * celldm(4) - celldm(5) * celldm(5));
    if (term < 0.) {
      Error("celldm does not make sense, check your data", iBravais);
    }
    term = sqrt(term / (1. - celldm(5) * celldm(5)));
    a1(0) = celldm(0);
    a2(0) = celldm(0) * celldm(1) * celldm(5);
    a2(1) = celldm(0) * celldm(1) * sigma;
    a3(0) = celldm(0) * celldm(2) * celldm(4);
    a3(1) =
        celldm(0) * celldm(2) * (celldm(3) - celldm(4) * celldm(5)) / sigma;
    a3(2) = celldm(0) * celldm(2) * term;

  } else {
    Error("nonexistent bravais lattice", iBravais);
  }

  if (iBravais != 0) {
    unitCell.col(0) = a1;
    unitCell.col(1) = a2;
    unitCell.col(2) = a3;
  }
}

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  if (delimiter == ' ') {
    for (std::string s2; tokenStream >> s2;) {
      tokens.push_back(s2);
    }
  } else {
    while (std::getline(tokenStream, token, delimiter)) {
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                  token.end());
      tokens.push_back(token);
    }
  }

  return tokens;
}

std::tuple<Crystal, PhononH0> QEParser::parsePhHarmonic(Context &context) {
  //  Here we read the dynamical matrix of inter-atomic force constants
  //	in real space.

  Kokkos::Profiling::pushRegion("parsePhHarmonic (QE)");

  std::string fileName = context.getPhFC2FileName();
  if (fileName.empty()) {
    Error("Must provide a FC2 file name");
  }

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open()) {
    Error("Dynamical matrix file not found");
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " + fileName + "." << std::endl;

  //  First line contains iBravais, celldm and other variables

  std::getline(infile, line);
  lineSplit = split(line, ' ');

  // we put a try catch here because if you
  // try to run a phonopy fc2.hdf5 file through this parser,
  // this is where things will fail
  int numElements = 0;
  try {
    numElements = std::stoi(lineSplit[0]);
  } catch (std::invalid_argument &) {
    Error("Something has gone wrong with parsing this qe.fc file.\n"
          "If you're trying to run phonon apps with a phonopy fc2.hdf5 file,\n"
          "make sure that you have set the phonopyDispFileName input variable.");
  }
  int numAtoms = std::stoi(lineSplit[1]);
  int iBravais = std::stoi(lineSplit[2]);

  Eigen::VectorXd celldm(6);
  celldm(0) = std::stod(lineSplit[3]);
  celldm(1) = std::stod(lineSplit[4]);
  celldm(2) = std::stod(lineSplit[5]);
  celldm(3) = std::stod(lineSplit[6]);
  celldm(4) = std::stod(lineSplit[7]);
  celldm(5) = std::stod(lineSplit[8]);

  Eigen::Matrix3d directUnitCell;
  if (iBravais == 0) {
    // In this case, unitCell is written in the file (in angstroms?)
    for (int i = 0; i < 3; i++) {
      std::getline(infile, line);
      lineSplit = split(line, ' ');
      for (int j = 0; j < 3; j++) {
        directUnitCell(j, i) = std::stod(lineSplit[j]);
      }
    }
  }

  // generate the unit cell vectors (also for iBravais != 0)
  getQELattice(iBravais, celldm, directUnitCell);

  //  Next, we read the atomic species
  std::vector<std::string> speciesNames;
  Eigen::VectorXd speciesMasses(numElements);
  for (int i = 0; i < numElements; i++) {
    std::getline(infile, line);
    lineSplit = split(line, '\'');
    speciesNames.push_back(lineSplit[1]);
    speciesMasses(i) = std::stod(lineSplit[2]);// in rydberg
  }

  //  we read the atomic positions
  // in the file, they appear in crystal coordinates
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    atomicSpecies(i) = std::stoi(lineSplit[1]) - 1;
    Eigen::Vector3d tmpVec;
    tmpVec(0) = std::stod(lineSplit[2]);
    tmpVec(1) = std::stod(lineSplit[3]);
    tmpVec(2) = std::stod(lineSplit[4]);
    // we convert from crystal to cartesian coordinates
    atomicPositions.row(i) = celldm(0) * tmpVec;
  }

  //  Read if hasDielectric
  std::getline(infile, line);
  line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
  bool hasDielectric = false;
  if (line == "T") {
    hasDielectric = true;
  }

  //	if there are the dielectric info, we can read dielectric matrix
  //	and the Born charges
  Eigen::Matrix3d dielectricMatrix = Eigen::Matrix3d::Zero();
  Eigen::Tensor<double, 3> bornCharges(numAtoms, 3, 3);
  bornCharges.setZero();

  if (hasDielectric) {
    for (int i : {0, 1, 2}) {
      std::getline(infile, line);
      lineSplit = split(line, ' ');
      for (int j : {0, 1, 2}) {
        dielectricMatrix(i, j) = std::stod(lineSplit[j]);
      }
    }

    for (int iAtom = 0; iAtom < numAtoms; iAtom++) {
      std::getline(infile, line);
      for (int i : {0, 1, 2}) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        for (int j : {0, 1, 2}) {
          bornCharges(iAtom, i, j) = std::stod(lineSplit[j]);
        }
      }
    }
  }

  //	Now we parse the coarse q grid
  std::getline(infile, line);
  lineSplit = split(line, ' ');
  Eigen::VectorXi qCoarseGrid(3);
  qCoarseGrid(0) = std::stoi(lineSplit[0]);
  qCoarseGrid(1) = std::stoi(lineSplit[1]);
  qCoarseGrid(2) = std::stoi(lineSplit[2]);

  Eigen::Tensor<double, 7> forceConstants(3, 3, qCoarseGrid[0], qCoarseGrid[1],
                                          qCoarseGrid[2], numAtoms, numAtoms);
  for (int ic : {0, 1, 2}) {
    int m1Test, m2Test, m3Test;
    double x;
    for (int jc : {0, 1, 2}) {
      for (int iat = 0; iat < numAtoms; iat++) {
        for (int jat = 0; jat < numAtoms; jat++) {
          // a line containing ic, jc, iat, jat
          std::getline(infile, line);

          for (int r3 = 0; r3 < qCoarseGrid[2]; r3++) {
            for (int r2 = 0; r2 < qCoarseGrid[1]; r2++) {
              for (int r1 = 0; r1 < qCoarseGrid[0]; r1++) {
                std::getline(infile, line);
                std::istringstream iss(line);
                iss >> m1Test >> m2Test >> m3Test >> x;
                forceConstants(ic, jc, r1, r2, r3, iat, jat) = x;
              }
            }
          }
        }
      }
    }
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
    std::cout << "Successfully parsed harmonic QE files.\n"
              << std::endl;
  }

  PhononH0 dynamicalMatrix(crystal, dielectricMatrix, bornCharges,
                           forceConstants, context.getSumRuleFC2());

  Kokkos::Profiling::popRegion();
  return std::make_tuple(crystal, dynamicalMatrix);
}

std::tuple<Crystal, ElectronH0Fourier>
QEParser::parseElHarmonicFourier(Context &context) {
  //  Here we read the XML file of quantum espresso.

  Kokkos::Profiling::pushRegion("parseElHarmonicFourier (QE)");

  std::string fileName = context.getElectronH0Name();
  double fourierCutoff = context.getElectronFourierCutoff();

  if (fileName.empty()) {
    Error("Must provide an XML file name");
  }

  std::vector<std::string> lineSplit;

  // load and parse XML file using a library
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(fileName.c_str());

  if (not result) {
    Error("Error parsing XML file");
  }
  if (mpi->mpiHead())
    std::cout << "Reading in " << fileName << "." << std::endl;

  pugi::xml_node output = doc.child("qes:espresso").child("output");

  // atomic species

  pugi::xml_node atomicSpeciesXML = output.child("atomic_species");
  int numElements = atomicSpeciesXML.attribute("ntyp").as_int();
  std::vector<std::string> speciesNames;
  Eigen::VectorXd speciesMasses(numElements);
  int i = 0;
  for (pugi::xml_node species : atomicSpeciesXML.children("species")) {
    speciesNames.emplace_back(species.attribute("name").value());
    speciesMasses(i) = species.child("mass").text().as_double();// in amu
    i += 1;
  }

  // atomic structure

  pugi::xml_node atomicStructure = output.child("atomic_structure");
  int numAtoms = atomicStructure.attribute("nat").as_int();

  //  we read the atomic positions

  pugi::xml_node atomicPositionsXML = atomicStructure.child("atomic_positions");
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  i = 0;
  int atomId = 0;
  std::string thisAtomName;
  for (pugi::xml_node atom : atomicPositionsXML.children("atom")) {
    thisAtomName = atom.attribute("name").value();
    // the XML doesn't describe atoms with a tag ID, but using names
    // here I find the index of the species in speciesNames, given the name
    auto itr =
        std::find(speciesNames.begin(), speciesNames.end(), thisAtomName);
    if (itr != speciesNames.cend()) {
      atomId = std::distance(speciesNames.begin(), itr);
    } else {
      Error("Element not found in XML");
    }
    atomicSpecies(i) = atomId;

    // note: atomic positions are in Cartesian coordinates in units of angstroms
    lineSplit = split(atom.child_value(), ' ');
    atomicPositions(i, 0) = std::stod(lineSplit[0]);
    atomicPositions(i, 1) = std::stod(lineSplit[1]);
    atomicPositions(i, 2) = std::stod(lineSplit[2]);
    i++;
  }

  // we read the unit cell

  Eigen::Matrix3d directUnitCell;
  Eigen::Vector3d thisValues;
  pugi::xml_node cell = atomicStructure.child("cell");
  lineSplit = split(cell.child_value("a1"), ' ');
  directUnitCell(0, 0) = std::stod(lineSplit[0]);
  directUnitCell(1, 0) = std::stod(lineSplit[1]);
  directUnitCell(2, 0) = std::stod(lineSplit[2]);
  lineSplit = split(cell.child_value("a2"), ' ');
  directUnitCell(0, 1) = std::stod(lineSplit[0]);
  directUnitCell(1, 1) = std::stod(lineSplit[1]);
  directUnitCell(2, 1) = std::stod(lineSplit[2]);
  lineSplit = split(cell.child_value("a3"), ' ');
  directUnitCell(0, 2) = std::stod(lineSplit[0]);
  directUnitCell(1, 2) = std::stod(lineSplit[1]);
  directUnitCell(2, 2) = std::stod(lineSplit[2]);

  // Now we parse the electronic structure

  pugi::xml_node bandStructureXML = output.child("band_structure");
  bool isLSDA = bandStructureXML.child("lsda").text().as_bool();
  bool nonCollinear = bandStructureXML.child("noncolin").text().as_bool();
  bool spinOrbit = bandStructureXML.child("spinorbit").text().as_bool();
  int numBands = bandStructureXML.child("nbnd").text().as_int();
  // note: nelec is written as double in the XML file!
  int numElectrons = int(bandStructureXML.child("nelec").text().as_double());

  // get fermi energy
  double homo;
  // it's an insulator
  if (bandStructureXML.child("highestOccupiedLevel")) {
    homo = bandStructureXML.child("highestOccupiedLevel").text().as_double();
  } else if (bandStructureXML.child("fermi_energy")) { // it's a metal or run with smearing
    homo = bandStructureXML.child("fermi_energy").text().as_double();
  } else {
    Error("Something is wrong, neither highestOccupiedLevel "
        "nor fermi_energy tags appear in XML file.");
  }
  homo *= 2.;// conversion from Hartree to Rydberg
  int numIrreduciblePoints = bandStructureXML.child("nks").text().as_int();

  pugi::xml_node startingKPoints = bandStructureXML.child("starting_k_points");
  // this may or may not be present! if so, I get mesh and offset
  pugi::xml_node mp = startingKPoints.child("monkhorst_pack");
  if (mp) {
    if(mpi->mpiHead()) {
        Error("Found Monkhorst-Pack mesh in XML file.\n"
        "This likely means you're reading in an SCF XML file, which\n"
        "doesn't make sense for Phoebe -- likely you forgot to perform NSCF on\n"
        "the full k-mesh first.");
    }
    // Error("Grid found in QE:XML, should have used full kPoints grid");
  }

  // Initialize the crystal class

  Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                  speciesNames, speciesMasses);
  crystal.print();

  // initialize reciprocal lattice cell
  // I need this to convert kPoints from cartesian to crystal coordinates

  pugi::xml_node basisSet = output.child("basis_set");
  pugi::xml_node recCell = basisSet.child("reciprocal_lattice");
  Eigen::Matrix3d bVectors;
  lineSplit = split(recCell.child_value("b1"), ' ');
  bVectors(0, 0) = std::stod(lineSplit[0]);
  bVectors(1, 0) = std::stod(lineSplit[1]);
  bVectors(2, 0) = std::stod(lineSplit[2]);
  lineSplit = split(recCell.child_value("b2"), ' ');
  bVectors(0, 1) = std::stod(lineSplit[0]);
  bVectors(1, 1) = std::stod(lineSplit[1]);
  bVectors(2, 1) = std::stod(lineSplit[2]);
  lineSplit = split(recCell.child_value("b3"), ' ');
  bVectors(0, 2) = std::stod(lineSplit[0]);
  bVectors(1, 2) = std::stod(lineSplit[1]);
  bVectors(2, 2) = std::stod(lineSplit[2]);

  // parse k-points and energies

  Eigen::Matrix<double, 3, Eigen::Dynamic> irrPoints(3, numIrreduciblePoints);
  Eigen::VectorXd irrWeights(numIrreduciblePoints);
  Eigen::MatrixXd irrEnergies(numIrreduciblePoints, numBands);
  Eigen::MatrixXd irrOccupations(numIrreduciblePoints, numBands);
  irrPoints.setZero();
  irrWeights.setZero();
  irrEnergies.setZero();
  irrOccupations.setZero();
  i = 0;
  for (pugi::xml_node kPoint : bandStructureXML.children("ks_energies")) {
    irrWeights(i) = kPoint.child("k_point").attribute("weight").as_double();
    lineSplit = split(kPoint.child_value("k_point"), ' ');

    // note:
    // k_cart = bVectors * k_crystal
    Eigen::Vector3d p;
    p(0) = std::stod(lineSplit[0]);
    p(1) = std::stod(lineSplit[1]);
    p(2) = std::stod(lineSplit[2]);

    // convert from cartesian to crystal coordinates
    p = bVectors.inverse() * p;
    irrPoints.col(i) = p;

    lineSplit = split(kPoint.child_value("eigenvalues"), ' ');
    for (int j = 0; j < numBands; j++) {
      irrEnergies(i, j) = std::stod(lineSplit[j]);
    }

    lineSplit = split(kPoint.child_value("occupations"), ' ');
    for (int j = 0; j < numBands; j++) {
      irrOccupations(i, j) = std::stod(lineSplit[j]);
    }

    i++;
  }

  // QE XML energies are in Hartree units. Must convert to rydberg
  irrEnergies *= 2.;

  // Now we do postprocessing

  if (isLSDA || nonCollinear) {
    Error("spin is not yet supported");
  }

  auto tup = Points::findMesh(irrPoints);
  auto mesh = std::get<0>(tup);
  auto offset = std::get<1>(tup);
  Points coarsePoints(crystal, mesh, offset);

  bool withVelocities = false;
  bool withEigenvectors = false;
  Particle particle(Particle::electron);
  FullBandStructure coarseBandStructure(numBands, particle, withVelocities,
                                        withEigenvectors, coarsePoints);
  // fill in the info on band structure
  for (int ik = 0; ik < numIrreduciblePoints; ik++) {
    // note: k-points in the XML files are not ordered in the same way
    // as in the scf.in file
    Eigen::Vector3d pointCoordinates = irrPoints.col(ik);
    Eigen::VectorXd thisEnergies = irrEnergies.row(ik);
    coarseBandStructure.setEnergies(pointCoordinates, thisEnergies);
  }

  context.setHasSpinOrbit(spinOrbit);
  if (spinOrbit) {
    numElectrons /= 2.;
  }
  context.setNumOccupiedStates(numElectrons);

  // if the user didn't set the Fermi level, we do it here.
  if (std::isnan(context.getFermiLevel()))
    context.setFermiLevel(homo);

  if (mpi->mpiHead())
    std::cout << "Done reading in " << fileName << "." << std::endl;

  ElectronH0Fourier electronH0(crystal, coarsePoints, coarseBandStructure,
                               fourierCutoff);

  Kokkos::Profiling::popRegion();
  return std::make_tuple(crystal, electronH0);
}

std::pair<Eigen::Tensor<double, 3>,
          Eigen::Tensor<double, 5>> parseWSVecFromWannier90(
    const std::string &fileName, const int &numWannier,
    const int &numVectors, const Eigen::Matrix3d &directUnitCell) {

  Kokkos::Profiling::pushRegion("parseWSVecFromWannier90");

  if (fileName.empty()) {
    Error("Must provide the Wannier90 WsVec file name");
  }

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open()) {
    Error("Wannier WsVec file not found");
  }

  //  First line contains the title and date, and the information whether
  // or not the Bravais vectors shifts are present
  std::getline(infile, line);
  {
    std::string s2 = ".true.";
    if (line.find(s2) == std::string::npos) { // string not found
     Error("Wannier90 didn't run with phase shifts. Remove wsVecFileName from input");
    }
  }

  Eigen::Tensor<double, 3> degeneracyShifts(numWannier, numWannier, numVectors);
  Eigen::Tensor<double, 5> phaseShifts(3, 8, numWannier, numWannier, numVectors);
  // note: 8 is the largest degeneracy possible for a BV vector in a 3D space
  // i.e. the corner of wigner-seitz cell is at most 8-fold degenerate
  degeneracyShifts.setZero();
  phaseShifts.setZero();

  std::vector<int> oldBV(3, 0);
  std::vector<int> thisBV(3, 0);
  std::vector<int> shift(3, 0);
  int iR = -1;
  while (std::getline(infile, line)) {
    auto lineSplit = split(line, ' ');
    thisBV[0] = std::stoi(lineSplit[0]);
    thisBV[1] = std::stoi(lineSplit[1]);
    thisBV[2] = std::stoi(lineSplit[2]);
    if (thisBV != oldBV) { iR++; oldBV = thisBV; }
    int iw1 = std::stoi(lineSplit[3]) - 1;
    int iw2 = std::stoi(lineSplit[4]) - 1;
    std::getline(infile, line);
    int thisDegeneracy = std::stoi(line);
    degeneracyShifts(iw1, iw2, iR) = thisDegeneracy;
    for (int iDeg = 0; iDeg < thisDegeneracy; ++iDeg) {
      std::getline(infile, line);
      auto lineSplit2 = split(line, ' ');
      Eigen::Vector3d tmpVector;
      tmpVector(0) = std::stoi(lineSplit2[0]);
      tmpVector(1) = std::stoi(lineSplit2[1]);
      tmpVector(2) = std::stoi(lineSplit2[2]);
      // the phase shifts are written in crystal coordinates
      // let's change them to cartesian coordinates
      tmpVector = directUnitCell * tmpVector;
      phaseShifts(0, iDeg, iw1, iw2, iR) = tmpVector(0);
      phaseShifts(1, iDeg, iw1, iw2, iR) = tmpVector(1);
      phaseShifts(2, iDeg, iw1, iw2, iR) = tmpVector(2);
    }
  }

  Kokkos::Profiling::popRegion();
  return std::make_pair(degeneracyShifts, phaseShifts);
}

std::tuple<Crystal, ElectronH0Wannier>
QEParser::parseElHarmonicWannier(Context &context, Crystal *inCrystal) {

  Kokkos::Profiling::pushRegion("parseElHarmonicWannier (QE)");

  //  Here we read the XML file of quantum espresso.

  std::string fileName = context.getElectronH0Name();

  if (fileName.empty()) {
    Error("Must provide the Wannier90 TB file name");
  }

  std::string line;
  std::vector<std::string> lineSplit;

  // open input file
  std::ifstream infile(fileName);

  if (not infile.is_open()) {
    Error("Wannier H0 file not found");
  }

  //  First line contains the title and date
  std::getline(infile, line);

  // Then, we have the directUnitCell of the crystal in angstroms
  Eigen::Matrix3d directUnitCell_(3, 3);
  directUnitCell_.setZero();
  for (int i = 0; i < 3; i++) {
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    for (int j = 0; j < 3; j++) {
      // unit cell is written in angstrom
      directUnitCell_(j, i) = std::stod(lineSplit[j]) / distanceBohrToAng;
    }
  }

  // Next, we the number of Wannier functions / bands, after disentanglement
  std::getline(infile, line);
  int numWannier = std::stoi(line);

  // The number of irreducible vectors in real space
  std::getline(infile, line);
  int numVectors = std::stoi(line);

  // now, we must read numVectors integers with the vector degeneracies
  // there can be only up to 15 numbers per line
  int numLines = numVectors / int(15);
  if (numVectors % 15 != 0) { numLines += 1; }

  // note: the file contains only phases to compute the FT of the hamiltonian
  // with the phase factor e^{ikR}
  // this does not have enough accuracy, and instead we will use the phase
  // factors e^{ik(R-R1+R2)}, shifted by the Wannier function centers
  // we parse the old phase factors anyway
  Eigen::VectorXd simpleVectorsDegeneracies(numVectors);
  simpleVectorsDegeneracies.setZero();
  {
    int j = 0;
    for (int i = 0; i < numLines; i++) {
      std::getline(infile, line);
      lineSplit = split(line, ' ');
      for (const auto &x : lineSplit) {
        int deg = std::stoi(x);
        simpleVectorsDegeneracies(j) = double(deg);
        j += 1;
      }
    }
  }

  // now we read the Hamiltonian in real space
  Eigen::MatrixXd bravaisVectors(3, numVectors);
  Eigen::Tensor<std::complex<double>, 3> h0R(numVectors, numWannier,
                                             numWannier);
  Eigen::Tensor<std::complex<double>, 4> rMatrix(3, numVectors, numWannier,
                                                 numWannier);
  bravaisVectors.setZero();
  h0R.setZero();
  rMatrix.setZero();

  // parse the Hamiltonian

  for (int iR = 0; iR < numVectors; iR++) {
    // first we have an empty line
    std::getline(infile, line);

    // then we read the lattice vector coordinates
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    bravaisVectors(0, iR) = std::stod(lineSplit[0]);
    bravaisVectors(1, iR) = std::stod(lineSplit[1]);
    bravaisVectors(2, iR) = std::stod(lineSplit[2]);

    for (int i = 0; i < numWannier; i++) {
      for (int j = 0; j < numWannier; j++) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        double re = std::stod(lineSplit[2]) / energyRyToEv;
        double im = std::stod(lineSplit[3]) / energyRyToEv;
        h0R(iR, i, j) = {re, im};// the matrix was in eV
      }
    }
  }

  // now parse the R matrix
  // the format is similar, but we have a complex vector

  for (int iR = 0; iR < numVectors; iR++) {
    // first we have an empty line
    std::getline(infile, line);

    // then we read the lattice vector coordinates
    std::getline(infile, line);
    lineSplit = split(line, ' ');
    // they have been initialized above, and they are the same

    for (int i = 0; i < numWannier; i++) {
      for (int j = 0; j < numWannier; j++) {
        std::getline(infile, line);
        lineSplit = split(line, ' ');
        double re = std::stod(lineSplit[2]) / distanceBohrToAng;
        double im = std::stod(lineSplit[3]) / distanceBohrToAng;
        rMatrix(0, iR, i, j) = {re, im};// the matrix was in eV
        re = std::stod(lineSplit[4]) / distanceBohrToAng;
        im = std::stod(lineSplit[5]) / distanceBohrToAng;
        rMatrix(1, iR, i, j) = {re, im};// the matrix was in eV
        re = std::stod(lineSplit[6]) / distanceBohrToAng;
        im = std::stod(lineSplit[7]) / distanceBohrToAng;
        rMatrix(2, iR, i, j) = {re, im};// the matrix was in eV
      }
    }
  }

  Eigen::Matrix3d directUnitCell(3, 3);
  if (inCrystal != nullptr) {
    directUnitCell = inCrystal->getDirectUnitCell();
  } else {
    directUnitCell = directUnitCell_;
  }

  // I need to convert crystalVectors in cartesian coordinates
  // must check if I am aligning the unit cell correctly
  bravaisVectors = directUnitCell * bravaisVectors;
  // note: for Wannier90, lattice vectors are the rows of the matrix

  ElectronH0Wannier electronH0(directUnitCell, bravaisVectors,
                               simpleVectorsDegeneracies, h0R, rMatrix);

  std::string wsVecFileName = context.getWsVecFileName();
  if (!wsVecFileName.empty()) {
    auto tt = parseWSVecFromWannier90(wsVecFileName, numWannier,
                                      numVectors, directUnitCell);
    auto degeneracyShifts = std::get<0>(tt);
    auto vectorsShifts = std::get<1>(tt);
    electronH0.addShiftedVectors(degeneracyShifts, vectorsShifts);
  }

  if (inCrystal != nullptr) {
    Kokkos::Profiling::popRegion();
    return std::make_tuple(*inCrystal, electronH0);
  } else {
    Eigen::MatrixXd atomicPositions = context.getInputAtomicPositions();
    Eigen::VectorXi atomicSpecies = context.getInputAtomicSpecies();
    std::vector<std::string> speciesNames = context.getInputSpeciesNames();
    // we default the masses to the conventional ones here.
    Eigen::VectorXd speciesMasses(speciesNames.size());
    PeriodicTable periodicTable;
    int i = 0;
    for (const auto &speciesName : speciesNames) {
      speciesMasses[i] = periodicTable.getMass(speciesName);
      i += 1;
    }

    Crystal crystal(context, directUnitCell, atomicPositions, atomicSpecies,
                    speciesNames, speciesMasses);
    crystal.print();
    Kokkos::Profiling::popRegion();
    return std::make_tuple(crystal, electronH0);
  }
}
